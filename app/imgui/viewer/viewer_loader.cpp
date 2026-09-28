#include "imgui/viewer/viewer_loader.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

#include "core/array_source.hpp"
#include "imgui/viewer/trace.hpp"
#include "imgui/viewer/viewer_constants.hpp"

namespace sirius::app::gui {

    namespace {
        // The z maximum projection and the exact value range of a (z, y, x)
        // volume in one pass. NaNs are skipped by both.
        void projectAndRange(const float* vol, Index nz, Index ny, Index nx, float* mip, float& lo, float& hi) {
            const Index n = ny * nx;
            lo = std::numeric_limits<float>::infinity();
            hi = -lo;
            std::fill_n(mip, n, -std::numeric_limits<float>::infinity());
            for (Index z = 0; z < nz; ++z) {
                const float* p = vol + z * n;
                for (Index i = 0; i < n; ++i) {
                    const float v = p[i];
                    if (std::isnan(v)) continue;
                    if (v > mip[i]) mip[i] = v;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            if (!(hi > lo)) {
                lo = std::isfinite(lo) ? lo : 0.0f;
                hi = lo + 1.0f;
            }
            for (Index i = 0; i < n; ++i)
                if (!std::isfinite(mip[i])) mip[i] = lo;
        }

        // One channel's volume as a brick of at most kVolumeTexelsMax texels
        // per axis: each texel averages a coarse sub-grid of its box, windowed
        // to 0..255. This is the loop that must not sit inside a frame.
        ReducedVolume reduceChannel(const ViewerLoader::Channel& ch) {
            ReducedVolume out;
            out.color = ch.color;
            if (!ch.data || ch.x <= 0 || ch.y <= 0 || ch.z <= 0) return out;
            const Index cap = viewer::kVolumeTexelsMax;
            const int fx = static_cast<int>((ch.x + cap - 1) / cap);
            const int fy = static_cast<int>((ch.y + cap - 1) / cap);
            const int fz = static_cast<int>((ch.z + cap - 1) / cap);
            out.tx = static_cast<int>((ch.x + fx - 1) / fx);
            out.ty = static_cast<int>((ch.y + fy - 1) / fy);
            out.tz = static_cast<int>((ch.z + fz - 1) / fz);
            out.texels.assign(static_cast<std::size_t>(out.tx) * static_cast<std::size_t>(out.ty) * static_cast<std::size_t>(out.tz), 0);
            const float scale = 255.0f / std::max(ch.hi - ch.lo, 1e-6f);
            const int sx = std::max(1, fx / 2), sy = std::max(1, fy / 2), sz = std::max(1, fz / 2);
            for (int z = 0; z < out.tz; ++z)
                for (int y = 0; y < out.ty; ++y)
                    for (int x = 0; x < out.tx; ++x) {
                        float acc = 0.0f;
                        int n = 0;
                        for (Index zz = static_cast<Index>(z) * fz; zz < std::min<Index>(static_cast<Index>(z + 1) * fz, ch.z); zz += sz)
                            for (Index yy = static_cast<Index>(y) * fy; yy < std::min<Index>(static_cast<Index>(y + 1) * fy, ch.y); yy += sy)
                                for (Index xx = static_cast<Index>(x) * fx; xx < std::min<Index>(static_cast<Index>(x + 1) * fx, ch.x); xx += sx) {
                                    acc += ch.data[(zz * ch.y + yy) * ch.x + xx];
                                    ++n;
                                }
                        const float v = n ? (acc / static_cast<float>(n) - ch.lo) * scale : 0.0f;
                        out.texels[(static_cast<std::size_t>(z) * static_cast<std::size_t>(out.ty) + static_cast<std::size_t>(y)) *
                                       static_cast<std::size_t>(out.tx) +
                                   static_cast<std::size_t>(x)] =
                            static_cast<unsigned char>(v > 255.0f ? 255 : (v > 0.0f ? static_cast<int>(v) : 0));
                    }
            return out;
        }
    } // namespace

    ViewerLoader::ViewerLoader(Poster post)
        : post_(std::move(post)), generation_(std::make_shared<std::atomic<std::uint64_t>>(1)),
          latestReduction_(std::make_shared<std::atomic<std::uint64_t>>(0)), alive_(std::make_shared<std::atomic<bool>>(true)) {
        thread_ = std::thread([this] { run(); });
    }

    ViewerLoader::~ViewerLoader() {
        // A running job sees the bump at its next check and returns; anything
        // still queued never starts. join() then guarantees no job touches
        // this object (or the data it holds) after the destructor, and the
        // cleared flag makes what was already posted to the GUI thread harmless.
        generation_->fetch_add(1);
        alive_->store(false);
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
            queue_.clear();
        }
        ready_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void ViewerLoader::enqueue(std::function<void()> job) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(job));
        }
        ready_.notify_one();
    }

    void ViewerLoader::run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return quit_ || !queue_.empty(); });
                if (quit_) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                job();
            } catch (...) {
                // a job reports its own errors; nothing may leave the thread
            }
        }
    }

    void ViewerLoader::cancelAll() {
        generation_->fetch_add(1);
        gen_ = generation_->load();
        {
            // what has not started never will: the generation check would
            // return at once, this only spares the wake-ups
            const std::lock_guard<std::mutex> lock(mutex_);
            queue_.clear();
        }
        pending_.clear();
        reductionPending_ = false;
        reductionKey_ = 0;
        latestReduction_->store(0);
    }

    bool ViewerLoader::pending(const std::shared_ptr<const StepOutput>& out, Index c, Index t) const {
        return pending_.count(Job{out.get(), c, t}) != 0;
    }

    bool ViewerLoader::prepare(const std::shared_ptr<const StepOutput>& out, Index c, Index t) {
        if (!out) return false;
        const Job job{out.get(), c, t};
        if (!pending_.insert(job).second) return false;   // already queued or running
        const std::uint64_t gen = gen_;
        auto generation = generation_;
        auto alive = alive_;
        Poster post = post_;
        ViewerLoader* self = this;
        enqueue([self, post, alive, generation, gen, out, c, t] {
            if (generation->load() != gen) return;   // the viewer moved on
            TraceClock clock;
            auto result = std::make_shared<Volume>();
            result->out = out;
            result->c = c;
            result->t = t;
            const Dims5& d = out->meta.dims;
            try {
                const float* vol = nullptr;
                if (out->array) {
                    vol = out->array->plane(c, t, 0);
                } else if (out->source) {
                    auto buf = std::make_shared<Buffer<float>>(Shape{d.z, d.y, d.x});
                    out->source->readVolume(c, t, buf->data(), [&](double f, const std::string& m) {
                        if (generation->load() != gen || !alive->load()) return;
                        post([self, alive, generation, gen, f, m] {
                            if (!alive->load() || generation->load() != gen) return;
                            if (self->volumeProgress) self->volumeProgress(0.9 * f, m);
                        });
                    });
                    vol = buf->data();
                    result->volume = std::move(buf);
                }
                if (generation->load() != gen) return;
                if (vol) {
                    auto mip = std::make_shared<Buffer<float>>(Shape{d.y, d.x});
                    projectAndRange(vol, d.z, d.y, d.x, mip->data(), result->lo, result->hi);
                    result->mip = std::move(mip);
                    result->ok = true;
                } else {
                    result->error = "no data source";
                }
            } catch (const std::exception& e) {
                result->ok = false;
                result->error = e.what();
            }
            result->micros = clock.micros();
            if (generation->load() != gen || !alive->load()) return;
            post([self, alive, generation, gen, result] {
                if (!alive->load() || generation->load() != gen) return;
                self->pending_.erase(Job{result->out.get(), result->c, result->t});
                if (self->volumeReady) self->volumeReady(*result);
            });
        });
        return true;
    }

    void ViewerLoader::reduce(std::uint64_t key, std::vector<Channel> channels) {
        if (key == reductionKey_ && reductionPending_) return;
        reductionKey_ = key;
        reductionPending_ = true;
        latestReduction_->store(key);
        const std::uint64_t gen = gen_;
        auto generation = generation_;
        auto latest = latestReduction_;
        auto alive = alive_;
        Poster post = post_;
        ViewerLoader* self = this;
        auto input = std::make_shared<std::vector<Channel>>(std::move(channels));
        enqueue([self, post, alive, generation, latest, gen, key, input] {
            if (generation->load() != gen) return;
            if (latest->load() != key) return;   // a newer request replaced this one before it started
            TraceClock clock;
            auto result = std::make_shared<Reduction>();
            result->key = key;
            for (const Channel& ch : *input) {
                if (generation->load() != gen) return;
                result->channels.push_back(reduceChannel(ch));
            }
            result->micros = clock.micros();
            if (generation->load() != gen || !alive->load()) return;
            post([self, alive, generation, gen, result] {
                if (!alive->load() || generation->load() != gen) return;
                if (self->reductionKey_ == result->key) self->reductionPending_ = false;
                if (self->reductionReady) self->reductionReady(*result);
            });
        });
    }

} // namespace sirius::app::gui
