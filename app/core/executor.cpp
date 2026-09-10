#include "core/executor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"

namespace sirius::app {

    const char* toString(StepReport::State s) noexcept {
        switch (s) {
            case StepReport::State::Running: return "running";
            case StepReport::State::Ran: return "ran";
            case StepReport::State::Cached: return "cached";
            case StepReport::State::Skipped: return "skipped";
            case StepReport::State::Failed: return "failed";
        }
        return "?";
    }

    namespace {
        // FNV-1a over the bytes: stable across runs (std::hash is not), cheap,
        // and collisions between the handful of steps of one session are not
        // a realistic concern.
        std::uint64_t fnv1a(const std::string& s) noexcept {
            std::uint64_t h = 1469598103934665603ull;
            for (unsigned char c : s) {
                h ^= c;
                h *= 1099511628211ull;
            }
            return h;
        }
    } // namespace

    std::string stableHash(const std::string& s) {
        char buf[24];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fnv1a(s)));
        return buf;
    }

    struct Executor::Entry {
        StepId id = 0;
        std::string fingerprint;
        CachePolicy policy = CachePolicy::Recompute;
        // Everything but the array stays in memory; the array is here for
        // Memory (and, until the next run, Recompute) and on disk for Disk.
        std::shared_ptr<const StepOutput> output;
        std::filesystem::path diskPath;
        std::size_t bytes = 0;
        bool arrayOnDisk = false;
        std::weak_ptr<const StepOutput> restored;   // the reloaded copy while someone holds it
        // The input labels the output's labels were shared from: a re-run
        // over the same input keeps the user's edits (editedLabelsOf).
        std::shared_ptr<const LabelVolume> labelsFrom;
    };

    Executor::Executor(std::filesystem::path scratchDir) : scratch_(std::move(scratchDir)) {
        std::error_code ec;
        std::filesystem::create_directories(scratch_, ec);
    }

    Executor::~Executor() {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [id, e] : entries_)
            if (e && !e->diskPath.empty()) {
                std::error_code ec;
                std::filesystem::remove(e->diskPath, ec);
            }
    }

    namespace {
        // The labels are (t, z, y, x) over the array's voxels, or they are
        // not this array's labels at all.
        bool labelsFit(const LabelVolume& labels, const Dims5& dims) noexcept {
            return labels.t() == dims.t && labels.z() == dims.z && labels.y() == dims.y && labels.x() == dims.x;
        }

        // What a parameter's path points at, not the path itself. An OTF, a
        // PSF, a flat-field image or the dataset can all be rewritten in place
        // while the pipeline still names the same file, and the step's output
        // would otherwise be served from the cache as though nothing had
        // changed. Size and modification time are enough to notice that, and
        // cost one stat.
        std::string fileStamp(const std::string& path) {
            if (path.empty()) return {};
            std::error_code ec;
            const std::filesystem::path file(path);
            const std::filesystem::file_status status = std::filesystem::status(file, ec);
            if (ec) return {};
            if (std::filesystem::is_regular_file(status)) {
                const std::uintmax_t size = std::filesystem::file_size(file, ec);
                if (ec) return {};
                const std::filesystem::file_time_type when = std::filesystem::last_write_time(file, ec);
                if (ec) return {};
                return std::to_string(size) + ":" + std::to_string(when.time_since_epoch().count());
            }
            if (std::filesystem::is_directory(status)) {
                // a zarr / N5 store or a folder dataset. The directory's own
                // stamp catches a file added, removed or renamed, but not a
                // chunk rewritten in place: walking a whole store on every
                // cache lookup would cost more than the run it protects.
                const std::filesystem::file_time_type when = std::filesystem::last_write_time(file, ec);
                if (ec) return {};
                return "dir:" + std::to_string(when.time_since_epoch().count());
            }
            return {};
        }
    } // namespace

    std::string Executor::fingerprint(const Pipeline& p, int index) const {
        std::string upstream;
        for (int i = 0; i <= index && i < p.size(); ++i) {
            const Step& s = p.at(i);
            if (i > 0 && !s.enabled) continue;   // a skipped step is transparent
            std::string own = s.kind + "|" + s.params.toJson().dump() + "|" + upstream;
            // every file the step reads, by identity rather than by name
            if (const Operation* op = findOperation(s.kind))
                for (const ParamSpec& spec : op->info().params) {
                    if (spec.type == ParamType::Path) {
                        const std::string stamp = fileStamp(s.params.getString(spec.key));
                        if (!stamp.empty()) own += "|" + spec.key + "@" + stamp;
                    } else if (spec.type == ParamType::StringList) {
                        // a list of files (the stitch tiles): each one that
                        // exists is stamped, a string that is no path costs a stat
                        for (const std::string& v : s.params.getStringList(spec.key)) {
                            const std::string stamp = fileStamp(v);
                            if (!stamp.empty()) own += "|" + spec.key + "@" + stamp;
                        }
                    }
                }
            upstream = stableHash(own);
        }
        return upstream;
    }

    std::shared_ptr<const StepOutput> Executor::load(Entry& e, PendingRestore& pending) const {
        if (!e.arrayOnDisk || !e.output) return e.output;
        // A spilled array is reloaded into a fresh StepOutput; the entry keeps
        // the on-disk copy so memory can be released again later. While a
        // caller (the viewer, a paint stroke) still holds the restored output
        // it is handed out again: one object, one read, instead of the whole
        // array coming off the disk for every refresh. The read itself is
        // not done here: this runs under the mutex, and a query must never
        // block a run (or the run a query) for gigabytes of I/O.
        if (auto held = e.restored.lock()) return held;
        pending.id = e.id;
        pending.shell = e.output;
        pending.path = e.diskPath;
        return nullptr;
    }

    std::shared_ptr<const StepOutput> Executor::restore(const PendingRestore& pending) const {
        if (!pending.shell) return nullptr;
        std::shared_ptr<Array5> array;
        try {
            array = readArrayFile(pending.path);
        } catch (const std::exception&) {
            // The spill file is gone or truncated (a temp cleaner, a full
            // disk): the entry is dropped so the step is recomputed instead
            // of every query throwing.
            std::lock_guard<std::mutex> g(mutex_);
            auto it = entries_.find(pending.id);
            if (it != entries_.end() && it->second && it->second->diskPath == pending.path) {
                Entry& e = *it->second;
                e.output = nullptr;
                e.arrayOnDisk = false;
                e.diskPath.clear();
                e.bytes = 0;
                e.restored.reset();
            }
            return nullptr;
        }
        auto restored = std::make_shared<StepOutput>(*pending.shell);
        restored->array = std::move(array);
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(pending.id);
        if (it != entries_.end() && it->second && it->second->diskPath == pending.path) {
            // another thread may have restored it meanwhile: one copy wins
            if (auto held = it->second->restored.lock()) return held;
            it->second->restored = restored;
        }
        return restored;
    }

    std::shared_ptr<const StepOutput> Executor::cached(const Pipeline& p, int index) const {
        if (index < 0 || index >= p.size()) return nullptr;
        const Step& s = p.at(index);
        const std::string fp = fingerprint(p, index);
        PendingRestore pending;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = entries_.find(s.id);
            if (it == entries_.end() || !it->second || it->second->fingerprint != fp) return nullptr;
            Entry& e = *it->second;
            if (!e.output) return nullptr;
            if (e.policy == CachePolicy::Recompute && !e.output->array && !e.output->source) return nullptr;
            if (auto out = load(e, pending)) return out;
        }
        return restore(pending);
    }

    std::shared_ptr<const StepOutput> Executor::lastOutput(StepId id) const {
        PendingRestore pending;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = entries_.find(id);
            if (it == entries_.end() || !it->second || !it->second->output) return nullptr;
            if (auto out = load(*it->second, pending)) return out;
        }
        return restore(pending);
    }

    std::shared_ptr<LabelVolume> Executor::lastLabels(StepId id) const {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second || !it->second->output) return nullptr;
        return it->second->output->labels;
    }

    void Executor::refreshPolicies(const Pipeline& p) {
        // Policies live in the pipeline and may change after an entry was
        // stored; the eviction pass below must see the current ones.
        for (auto& [id, e] : entries_)
            if (e)
                if (const Step* s = p.find(id)) e->policy = s->cache;
    }

    std::shared_ptr<LabelVolume> Executor::editedLabelsOf(StepId id, const std::shared_ptr<const LabelVolume>& from) const {
        if (!from) return nullptr;
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second || !it->second->output) return nullptr;
        const Entry& e = *it->second;
        if (!e.output->labels || !e.output->labels->edited() || e.labelsFrom != from) return nullptr;
        return e.output->labels;
    }

    void Executor::store(const Step& step, const std::string& fp, std::shared_ptr<const StepOutput> out,
                         std::shared_ptr<const LabelVolume> labelsFrom) {
        // A disk spill is written before the lock is taken (the file can be
        // gigabytes, and queries must not wait for it); the name is unique
        // per store so it never races an earlier file of the same step.
        std::filesystem::path diskPath;
        const std::size_t bytes = out && out->array ? out->array->bytes() : 0;
        if (step.cache == CachePolicy::Disk && out && out->array && !out->array->empty()) {
            diskPath = scratch_ / ("step-" + std::to_string(step.id) + "-" + fp + "-" + std::to_string(++spillCounter_) + ".sir5");
            try {
                writeArrayFile(diskPath, *out->array);
            } catch (...) {
                std::error_code ec;
                std::filesystem::remove(diskPath, ec);
                throw;
            }
            if (spillObserver_) spillObserver_(diskPath);
            auto shell = std::make_shared<StepOutput>(*out);
            shell->array = nullptr;
            out = shell;
        }
        std::filesystem::path stale;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto& slot = entries_[step.id];
            if (!slot) slot = std::make_unique<Entry>();
            Entry& e = *slot;
            stale = std::move(e.diskPath);
            e.id = step.id;
            e.fingerprint = fp;
            e.policy = step.cache;
            e.bytes = bytes;
            e.diskPath = std::move(diskPath);
            e.arrayOnDisk = !e.diskPath.empty();
            e.output = std::move(out);
            e.labelsFrom = std::move(labelsFrom);
            e.restored.reset();
            // "Recompute" keeps nothing beyond the most recent result: drop the
            // arrays of every other recompute entry now that a newer one exists.
            evictRecomputeExcept(step.id);
        }
        if (!stale.empty()) {
            std::error_code ec;
            std::filesystem::remove(stale, ec);
        }
    }

    void Executor::evictRecomputeExcept(StepId keep) {
        for (auto& [id, other] : entries_) {
            if (!other || id == keep || other->policy != CachePolicy::Recompute || !other->output) continue;
            if (other->output->array) {
                auto shell = std::make_shared<StepOutput>(*other->output);
                shell->array = nullptr;
                other->output = shell;
                other->bytes = 0;
            }
        }
    }

    std::shared_ptr<const StepOutput> Executor::run(const Pipeline& p, int index, const StepContext& ctx,
                                                    std::vector<StepReport>* reports,
                                                    const std::function<void(const StepReport&)>& onStep) {
        if (index < 0 || index >= p.size()) throw std::out_of_range("Executor::run: no step " + std::to_string(index));
        // Start from the nearest fresh output at or below the target rather
        // than from the top. A step is fresh by its fingerprint, whether or
        // not an upstream Recompute entry still holds its array: a fresh
        // target is served as it is instead of the upstream (and then, its
        // array evicted by the upstream's store, the target) running again.
        std::shared_ptr<const StepOutput> current;
        int start = -1;
        for (int i = index; i >= 0; --i) {
            if (i > 0 && !p.at(i).enabled) continue;
            if (auto c = cached(p, i)) {
                current = std::move(c);
                start = i;
                break;
            }
        }
        for (int i = 0; i <= index; ++i) {
            const Step& step = p.at(i);
            StepReport report;
            report.id = step.id;
            report.index = i;
            if (i > 0 && !step.enabled) {
                report.state = StepReport::State::Skipped;
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                continue;
            }
            if (i <= start) {
                // upstream of the output the run continues from: nothing to do
                report.state = StepReport::State::Cached;
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                continue;
            }
            const std::string fp = fingerprint(p, i);
            if (auto c = cached(p, i)) {
                current = std::move(c);
                report.state = StepReport::State::Cached;
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                continue;
            }
            report.state = StepReport::State::Running;
            if (onStep) onStep(report);
            if (ctx.isCancelled()) throw CancelledError();

            const Operation& op = step.op();
            StepInput input;
            if (i > 0) {
                if (!current) throw std::runtime_error("step " + Step::number(i) + " has no input");
                input = current->asInput();
            }
            const auto t0 = std::chrono::steady_clock::now();
            std::shared_ptr<StepOutput> out;
            try {
                out = std::make_shared<StepOutput>(op.run(input, step.params, ctx));
            } catch (const std::exception& e) {
                if (isCancellation(e)) throw CancelledError();   // not the step's fault
                report.state = StepReport::State::Failed;
                report.error = e.what();
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                throw std::runtime_error("Step " + Step::number(i) + " " + step.name + ": " + e.what());
            }
            const auto t1 = std::chrono::steady_clock::now();
            out->seconds = std::chrono::duration<double>(t1 - t0).count();
            if (out->meta.dims.numel() <= 0) out->meta.dims = input.meta.dims;
            // Labels carried through unless the step produced its own: a
            // volume of this step's own over the input's voxels, copied on
            // the first edit, so painting here never touches the upstream cache.
            // Only onto the same grid: a step that resamples, deskews or
            // reduces an axis leaves its labels null on purpose, and the
            // input's labels indexed with the new dims would read past their
            // buffer in every consumer (crop, export, cleanup).
            std::shared_ptr<const LabelVolume> labelsFrom;
            if (!out->labels && input.labels && labelsFit(*input.labels, out->meta.dims)) {
                // The user's corrections on this step survive its re-run (a
                // parameter change, a Recompute eviction) as long as they were
                // made over the labels it is about to carry through again; an
                // upstream that re-segmented supersedes them.
                labelsFrom = input.labels;
                std::shared_ptr<LabelVolume> kept = editedLabelsOf(step.id, labelsFrom);
                out->labels = kept ? kept : input.labels->share();
            }
            {
                std::lock_guard<std::mutex> g(mutex_);
                refreshPolicies(p);
            }
            store(step, fp, out, labelsFrom);
            current = out;
            report.state = StepReport::State::Ran;
            report.seconds = out->seconds;
            report.note = out->note;
            if (reports) reports->push_back(report);
            if (onStep) onStep(report);
        }
        return current;
    }

    std::shared_ptr<const StepOutput> Executor::runAll(const Pipeline& p, const StepContext& ctx,
                                                       std::vector<StepReport>* reports,
                                                       const std::function<void(const StepReport&)>& onStep) {
        return run(p, p.size() - 1, ctx, reports, onStep);
    }

    void Executor::seed(const Pipeline& p, int index, std::shared_ptr<const StepOutput> out) {
        if (index < 0 || index >= p.size()) return;
        {
            std::lock_guard<std::mutex> g(mutex_);
            refreshPolicies(p);
        }
        store(p.at(index), fingerprint(p, index), std::move(out));
    }

    void Executor::markStale(StepId id) {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it != entries_.end() && it->second) it->second->fingerprint.clear();
    }

    void Executor::invalidate(StepId id) {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end()) return;
        if (it->second && !it->second->diskPath.empty()) {
            std::error_code ec;
            std::filesystem::remove(it->second->diskPath, ec);
        }
        entries_.erase(it);
    }

    void Executor::clear() {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [id, e] : entries_)
            if (e && !e->diskPath.empty()) {
                std::error_code ec;
                std::filesystem::remove(e->diskPath, ec);
            }
        entries_.clear();
    }

    std::size_t Executor::cachedBytes() const {
        std::lock_guard<std::mutex> g(mutex_);
        std::size_t total = 0;
        std::vector<const std::uint32_t*> counted;   // label voxels shared by several steps count once
        for (const auto& [id, e] : entries_)
            if (e && e->output) {
                if (e->output->array) total += e->output->array->bytes();
                const LabelVolume* labels = e->output->labels.get();
                if (!labels || labels->empty()) continue;
                const std::uint32_t* voxels = labels->view().data();
                if (std::find(counted.begin(), counted.end(), voxels) != counted.end()) continue;
                counted.push_back(voxels);
                total += static_cast<std::size_t>(labels->t() * labels->volumeSize()) * sizeof(std::uint32_t);
            }
        return total;
    }

    std::size_t Executor::cachedBytesOf(StepId id) const {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second) return 0;
        return it->second->bytes;
    }

    // --- spill files --------------------------------------------------------------

    namespace {
        constexpr char kMagic[8] = {'S', 'I', 'R', '5', 'A', 'R', 'R', '1'};
    }

    void Executor::writeArrayFile(const std::filesystem::path& path, const Array5& a) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write cache file " + path.string());
        out.write(kMagic, sizeof kMagic);
        const Index dims[5] = {a.dims().c, a.dims().t, a.dims().z, a.dims().y, a.dims().x};
        out.write(reinterpret_cast<const char*>(dims), sizeof dims);
        out.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.bytes()));
        if (!out) throw std::runtime_error("short write to cache file " + path.string());
    }

    std::shared_ptr<Array5> Executor::readArrayFile(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read cache file " + path.string());
        char magic[8];
        in.read(magic, sizeof magic);
        if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("not a cache file: " + path.string());
        Index dims[5];
        in.read(reinterpret_cast<char*>(dims), sizeof dims);
        if (!in) throw std::runtime_error("short read from cache file " + path.string());
        // The extents size the allocation below: a truncated or foreign file
        // must not turn into a huge or negative request (Dims5::numel checks
        // the product; the sign is checked here).
        for (Index d : dims)
            if (d <= 0) throw std::runtime_error("corrupt cache file (bad extents): " + path.string());
        auto a = std::make_shared<Array5>(Dims5{dims[0], dims[1], dims[2], dims[3], dims[4]});
        in.read(reinterpret_cast<char*>(a->data()), static_cast<std::streamsize>(a->bytes()));
        if (!in) throw std::runtime_error("short read from cache file " + path.string());
        return a;
    }

} // namespace sirius::app
