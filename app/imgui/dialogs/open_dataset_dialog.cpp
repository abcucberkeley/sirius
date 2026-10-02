// File ▸ Open dataset…: path + Browse, the facts the file reports, and --
// when the metadata does not settle them -- how the pages map onto
// (c, t, z), the voxel size, the channel names and the raw SIM layout.
// A recent-files table sits below for one-click reopening. "Folder…" picks a
// folder of TIFF files: one with a manifest (core/manifest.hpp) opens like a
// file; one without goes through the folder dialog, which opens the dataset
// itself.
//
// The probe reads headers only, but on a network drive even that takes its
// time: it runs on the dialog's own thread, 250 ms after the last edit of
// the path, and what it found is taken only when the path is still the one
// it was started for.
#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

#include "core/array_source.hpp"
#include "core/manifest.hpp"
#include "core/remote_source.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/open_dataset_common.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {
        using namespace dataset_dialogs;
        using theme::px;
        namespace fs = std::filesystem;
        using Clock = std::chrono::steady_clock;

        constexpr int kProbeDelayMs = 250;
        const char* const kOrders[] = {"czt", "ctz", "zct", "ztc", "tcz", "tzc"};

        std::vector<platform::FileFilter> fileFilters() {
            // "*.ome.tif" is "*.tif" to a native dialog: the last part, once
            std::vector<std::string> exts;
            for (const std::string& e : readableExtensions()) {
                const std::size_t dot = e.find_last_of('.');
                const std::string last = toLower(dot == std::string::npos ? e : e.substr(dot + 1));
                if (!last.empty() && std::find(exts.begin(), exts.end(), last) == exts.end()) exts.push_back(last);
            }
            if (exts.empty()) exts = {"tif", "tiff"};
            return {{"Datasets", join(exts, ",")}, {"SIRIUS dataset", "toml"}, {"All files", "*"}};
        }

        // "stack.ome.tif" -> "tif"
        std::string suffixOf(const std::string& name) {
            const std::size_t dot = name.find_last_of('.');
            return dot == std::string::npos || dot == 0 ? std::string() : name.substr(dot + 1);
        }

        // "Sep 3 14:05"
        std::string modifiedText(const fs::path& p) {
            std::error_code ec;
            const fs::file_time_type ft = fs::last_write_time(p, ec);
            if (ec) return {};
            // C++17 has no clock_cast: the file clock's distance from now, applied to the system clock
            const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            const std::time_t t = std::chrono::system_clock::to_time_t(sys);
            std::tm tm{};
#ifdef _WIN32
            if (localtime_s(&tm, &t) != 0) return {};
#else
            if (!localtime_r(&t, &tm)) return {};
#endif
            static const char* const months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
            return format("%s %d %02d:%02d", months[std::clamp(tm.tm_mon, 0, 11)], tm.tm_mday, tm.tm_hour, tm.tm_min);
        }

        // The whole of `s` as a number, nothing left over.
        bool toNumber(const std::string& s, double& out) {
            if (s.empty()) return false;
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (!end || end == s.c_str() || *end != '\0') return false;
            out = v;
            return true;
        }

        // --- what the thread reports --------------------------------------------------
        struct ProbeResult {
            enum class Kind { Empty,
                              Missing,
                              BareFolder,   // TIFF files, no manifest
                              Probed,
                              Failed };
            std::uint64_t generation = 0;
            Kind kind = Kind::Empty;
            int tiffs = 0;
            bool folder = false;      // a folder with a manifest, or a manifest file
            DatasetMeta meta;
            std::string error;
        };

        ProbeResult probePath(const std::string& p, std::uint64_t generation) {
            ProbeResult r;
            r.generation = generation;
            if (p.empty()) return r;
            try {
                if (isRemoteDatasetPath(p)) {
                    // on the cluster: the worker there reads the header (core/remote_source.hpp)
                    r.meta = probeDataset(p);
                    r.kind = ProbeResult::Kind::Probed;
                    return r;
                }
                std::error_code ec;
                const fs::path path = toPath(p);
                if (!fs::exists(path, ec)) {
                    r.kind = ProbeResult::Kind::Missing;
                    return r;
                }
                r.folder = isManifestDataset(p);
                if (!r.folder && fs::is_directory(path, ec)) {
                    // a folder of TIFFs without a manifest is not yet a dataset
                    r.tiffs = static_cast<int>(tiffNamesInOrder(path).size());
                    if (r.tiffs > 0) {
                        r.kind = ProbeResult::Kind::BareFolder;
                        return r;
                    }
                }
                r.meta = probeDataset(p);
                r.kind = ProbeResult::Kind::Probed;
            } catch (const std::exception& e) {
                r.kind = ProbeResult::Kind::Failed;
                r.error = e.what();
            }
            return r;
        }

        struct RecentRow {
            std::string path;
            std::string name;
            std::string format;
            std::string modified;
        };

        struct OneStackResult {
            std::string folder;
            DatasetMeta first;
            bool firstProbed = false;
            DatasetManifest manifest;
            std::string error;
        };

        class OpenDatasetDialog final : public Dialog {
        public:
            using Accepted = std::function<void(const std::string&, const OpenOptions&)>;

            OpenDatasetDialog(App& app, const std::string& initialPath, Accepted accepted)
                : accepted_(std::move(accepted)), path_(initialPath), worker_(app.bridge()), scanWorker_(app.bridge()) {
                // The names at once; what needs the disk (a folder with a
                // manifest, the date, whether the file is still there) when
                // the thread has looked. That thread is not the probe's: one
                // recent entry on an unreachable share keeps it for as long as
                // the network takes, and the path being opened must not wait.
                for (const std::string& f : App::recentFiles()) {
                    RecentRow row;
                    row.path = f;
                    row.name = fileName(f);
                    row.format = suffixOf(row.name).empty() ? std::string("dir") : suffixOf(row.name);
                    recent_.push_back(std::move(row));
                }
                if (!recent_.empty()) {
                    std::vector<RecentRow> rows = recent_;
                    scanWorker_.run([this, alive = alive_, rows](const Worker::Post& post) mutable {
                        for (RecentRow& row : rows) {
                            if (!alive->load()) return;
                            try {
                                if (isRemoteDatasetPath(row.path)) {
                                    row.format = "cluster";
                                    continue;
                                }
                                std::error_code ec;
                                const fs::path p = toPath(row.path);
                                if (!fs::exists(p, ec)) {
                                    row.modified = "missing";
                                    continue;
                                }
                                if (fs::is_directory(p, ec) && isFolderDataset(row.path)) row.format = "folder";
                                row.modified = modifiedText(p);
                            } catch (const std::exception&) {
                                row.modified = "missing";
                            }
                        }
                        post([this, alive, rows] {
                            if (alive->load() && isOpen()) recent_ = rows;
                        });
                    });
                }
                scheduleProbe();
            }

            ~OpenDatasetDialog() override { alive_->store(false); }

            std::string title() const override { return "Open dataset"; }
            ImVec2 size() const override { return ImVec2(640, 0); }

            void draw(App& app) override {
                popupAtStart_ = popupAbove();
                if (probeAt_) {
                    if (Clock::now() >= *probeAt_) startProbe();
                    else app.requestRedraw();   // the timer: frames until it is due
                }
                drawPathRow(app);
                drawFacts();
                if (showLayout_) drawLayout();
                drawReadAs();
                drawRecent();
                drawButtons(app);
            }

            // After the frame the dialog left the screen in: what the caller
            // does with the path (a box, the next dialog) is not drawn inside this one.
            void closed(App&) override {
                if (acceptedNow_ && accepted_) accepted_(acceptedPath_, acceptedOptions_);
            }

            // The close box or Escape: the same as Cancel.
            bool canClose(App&) override {
                forgetAccept();
                return true;
            }

        private:
            // --- the path and its probe ----------------------------------------------

            std::string path() const { return trimmed(path_); }

            void setPath(const std::string& p) {
                if (p == path_) return;
                path_ = p;
                pathChanged();
            }

            void pathChanged() {
                // What was probed describes the previous path: nothing opens
                // until this one is probed, or a recent row double-clicked within
                // the probe's delay opened the new file with the old one's layout.
                probeOk_ = false;
                openEnabled_ = false;
                acceptPending_ = false;
                // Open as one stack writes a manifest into the folder: not
                // into this one before its probe has said it is a bare folder.
                oneStackVisible_ = false;
                pageCheck_.clear();
                scheduleProbe();
            }

            void scheduleProbe() {
                ++generation_;   // a probe still running is for a path that was
                latestProbe_->store(generation_);
                probing_ = false;
                probeAt_ = Clock::now() + std::chrono::milliseconds(kProbeDelayMs);
            }

            void startProbe() {
                probeAt_.reset();
                probing_ = true;
                const std::uint64_t generation = ++generation_;
                latestProbe_->store(generation);
                worker_.run([this, alive = alive_, latest = latestProbe_, p = path(), generation](const Worker::Post& post) {
                    // A probe for a path that has been replaced since reads
                    // nothing: the newer one is queued or scheduled, and
                    // clears `probing_` when it reports.
                    if (!alive->load() || latest->load() != generation) return;
                    ProbeResult r = probePath(p, generation);
                    post([this, alive, r = std::move(r)] {
                        if (alive->load() && isOpen()) applyProbe(r);
                    });
                });
            }

            // The facts, the page layout and the metadata fields for the path
            // as it is now.
            void applyProbe(const ProbeResult& r) {
                if (r.generation != generation_) return;   // the path has changed since
                probing_ = false;
                probeOk_ = false;
                isFolder_ = false;
                error_.clear();
                oneStackVisible_ = false;
                showLayout_ = true;
                openEnabled_ = false;
                switch (r.kind) {
                    case ProbeResult::Kind::Empty:
                        facts_ = "Choose a TIFF / OME-TIFF file, a zarr / N5 store or a folder of TIFF files.";
                        break;
                    case ProbeResult::Kind::Missing: facts_ = "No such file or directory."; break;
                    case ProbeResult::Kind::BareFolder:
                        facts_ = format("Folder of %d TIFF file(s) without a %s manifest. "
                                        "Open as one stack reads them in name order, one time point each. "
                                        "For channels, tiles or another order, use Folder….",
                                        r.tiffs, DatasetManifest::kFileName);
                        oneStackVisible_ = true;
                        break;
                    case ProbeResult::Kind::Failed:
                        facts_.clear();
                        error_ = r.error;
                        break;
                    case ProbeResult::Kind::Probed: applyMeta(r); break;
                }
                updatePageCheck();
                if (acceptPending_) {
                    acceptPending_ = false;
                    if (probeOk_ && openEnabled_) finish();
                }
            }

            void applyMeta(const ProbeResult& r) {
                probed_ = r.meta;
                probeOk_ = true;
                isFolder_ = r.folder;
                showLayout_ = !r.folder;   // the manifest settles layout, voxel size and channels
                const DatasetMeta& m = probed_;
                pages_ = m.dims.planes();
                dimsFromMetadata_ = r.folder || m.format != "tiff";   // plain TIFF: the page mapping is the user's call
                // An OME-TIFF's pages are in its DimensionOrder, which the
                // first order stands for (options() passes any other): an
                // order chosen for the file before is not this file's.
                if (m.format == "ome-tiff") order_ = 0;
                facts_ = format("%s · %s · %s · %s · %d channel(s)", m.format.c_str(), m.shapeString().c_str(), toString(m.sourceType),
                                bytesText(m.bytesOnDisk).c_str(), static_cast<int>(m.channels.size()));
                if (m.hasTiles()) facts_ += format(" · %d tiles", static_cast<int>(m.tiles.size()));
                // the file's own counts, not the spin boxes' usual ranges: a
                // 100-channel stack is 100 channels (drawLayout widens the ranges)
                c_ = std::max<std::int64_t>(m.dims.c, 1);
                t_ = std::max<std::int64_t>(m.dims.t, 1);
                z_ = std::max<std::int64_t>(m.dims.z, 0);
                if (m.voxelUm[0] > 0.0) voxel_[0] = m.voxelUm[0];
                if (m.voxelUm[1] > 0.0) voxel_[1] = m.voxelUm[1];
                if (m.voxelUm[2] > 0.0) voxel_[2] = m.voxelUm[2];
                std::vector<std::string> names;
                for (const ChannelInfo& ch : m.channels) {
                    std::string n = ch.label;
                    if (ch.wavelengthNm > 0) n = std::to_string(static_cast<int>(std::lround(ch.wavelengthNm))) + " " + n;
                    names.push_back(n);
                }
                channels_ = join(names, ", ");
                probedChannels_ = channels_;
                sim_.present = m.sim.present;
                sim_.dirs = std::clamp(m.sim.ndirs, 1, 9);
                sim_.phases = std::clamp(m.sim.nphases, 1, 15);
                sim_.fastSi = m.sim.fastSi;
                openEnabled_ = true;
            }

            void updatePageCheck() {
                if (!probeOk_) {
                    pageCheck_.clear();
                    return;
                }
                if (isFolder_) {
                    pageCheck_.clear();
                    openEnabled_ = true;
                    return;
                }
                const Index c = c_, t = t_;
                Index z = z_;
                const Index pages = pages_;
                if (z == 0) z = (c * t > 0 && pages % (c * t) == 0) ? pages / (c * t) : 0;
                const bool ok = z > 0 && c * t * z == pages;
                pageCheck_ = ok ? format("%lld pages = c%lld × t%lld × z%lld", static_cast<long long>(pages), static_cast<long long>(c),
                                         static_cast<long long>(t), static_cast<long long>(z))
                                : format("%lld pages do not divide into c%lld × t%lld × z%lld", static_cast<long long>(pages),
                                         static_cast<long long>(c), static_cast<long long>(t), static_cast<long long>(z_));
                pageCheckOk_ = ok;
                openEnabled_ = ok;
            }

            // --- accepting ---------------------------------------------------------------

            void accept() {
                // A path picked or typed a moment ago is probed now, so the
                // options that go with it are its own; one that does not open
                // is not accepted (the Open button is disabled for it too).
                if (probeAt_ || probing_) {
                    acceptPending_ = true;
                    if (probeAt_) startProbe();
                    return;
                }
                if (!probeOk_ || !openEnabled_) return;
                finish();
            }

            void finish() {
                acceptedNow_ = true;
                acceptedPath_ = path();
                acceptedOptions_ = options();
                close();
            }

            // Cancelled: an Open that is waiting for the probe does not happen,
            // nor one the probe completed at the start of this frame, before
            // the Cancel of this frame was seen.
            void forgetAccept() { acceptPending_ = acceptedNow_ = false; }

            void cancel() {
                forgetAccept();
                close();
            }

            OpenOptions options() const {
                OpenOptions o;
                if (isFolder_) {
                    // everything else comes from the manifest; start on the first tile
                    o.readAll = readAs_ == 1;
                    o.tile = 0;
                    return o;
                }
                PageOrder po;
                po.order = kOrders[std::clamp(order_, 0, static_cast<int>(std::size(kOrders)) - 1)];
                po.c = static_cast<Index>(c_);
                po.t = static_cast<Index>(t_);
                po.z = static_cast<Index>(z_);
                const DatasetMeta& m = probed_;
                // An order other than the default is the user's even where the
                // metadata gives the dimensions: an OME DimensionOrder can be
                // wrong. Only a TIFF's pages have an order to override.
                const bool paged = m.format == "tiff" || m.format == "ome-tiff";
                if (!dimsFromMetadata_ || (paged && po.order != kOrders[0]) || po.c != m.dims.c || po.t != m.dims.t ||
                    (po.z != 0 && po.z != m.dims.z))
                    o.pageOrder = po;
                if (voxel_ != m.voxelUm) o.voxelUm = voxel_;
                // channels: "488 name, 640 other". The text does not carry all
                // of a channel (its colour, its exposure) and does not read
                // back exactly (a label "488" reads as a wavelength), so the
                // file's channels are replaced only when the field was edited,
                // and then only the parts that were.
                if (trimmed(channels_) != trimmed(probedChannels_)) {
                    const std::vector<std::string> parts = split(channels_, ',', true);
                    const std::vector<std::string> probedParts = split(probedChannels_, ',', true);
                    std::vector<ChannelInfo> channels;
                    for (std::size_t i = 0; i < parts.size(); ++i) {
                        const bool known = i < m.channels.size();
                        if (known && i < probedParts.size() && trimmed(parts[i]) == trimmed(probedParts[i])) {
                            channels.push_back(m.channels[i]);
                            continue;
                        }
                        ChannelInfo ch = known ? m.channels[i] : ChannelInfo{};
                        std::string s = trimmed(parts[i]);
                        const std::size_t space = s.find(' ');
                        const std::string head = space == std::string::npos ? s : s.substr(0, space);
                        double nm = 0.0;
                        if (toNumber(head, nm) && nm > 100.0) s = space == std::string::npos ? std::string() : trimmed(s.substr(space + 1));
                        else nm = 0.0;
                        ch.label = s;
                        // a new wavelength brings its colour; the same one keeps the file's
                        if (!known || std::abs(nm - ch.wavelengthNm) >= 0.5) {
                            ch.wavelengthNm = nm;
                            ch.color = colorForWavelength(nm);
                        }
                        channels.push_back(ch);
                    }
                    bool sameChannels = channels.size() == m.channels.size();
                    for (std::size_t i = 0; sameChannels && i < channels.size(); ++i)
                        sameChannels = channels[i].label == m.channels[i].label &&
                                       std::abs(channels[i].wavelengthNm - m.channels[i].wavelengthNm) < 0.5;
                    if (!channels.empty() && !sameChannels) o.channels = channels;
                }
                SimLayout sim;
                sim.present = sim_.present;
                sim.ndirs = static_cast<int>(sim_.dirs);
                sim.nphases = static_cast<int>(sim_.phases);
                sim.fastSi = sim_.fastSi;
                if (sim.present != m.sim.present || sim.ndirs != m.sim.ndirs || sim.nphases != m.sim.nphases || sim.fastSi != m.sim.fastSi)
                    o.sim = sim;
                o.readAll = readAs_ == 1;
                return o;
            }

            // --- a folder of TIFFs as one stack ----------------------------------------------

            // A folder of TIFFs with no manifest, read the plain way: every file
            // one time point of one stack, in name order. The manifest is
            // written beside the files, as the Folder dialog does, so the folder
            // opens directly from then on and the reading can be corrected by hand.
            void openAsOneStack(App& app) {
                const std::string folder = path();
                // only for the path the probe found to be a bare folder of TIFFs
                if (folder.empty() || oneStackBusy_ || probeAt_ || probing_ || !oneStackVisible_) return;
                // The manifest is written before the dataset is opened, so the
                // refusal has to come first: otherwise a run in progress leaves
                // the file behind for a dataset that never opened.
                if (!app.wb().canEdit()) {
                    app.message("Open as one stack", "A run is in progress: cancel it (Esc) or wait before opening a dataset.",
                                MessageIcon::Info);
                    return;
                }
                oneStackBusy_ = true;
                worker_.run([this, alive = alive_, folder, appPtr = &app](const Worker::Post& post) {
                    OneStackResult r;
                    r.folder = folder;
                    try {
                        const fs::path dir = toPath(folder);
                        // The manifest written below would replace one the
                        // folder has, a hand-edited one included.
                        if (isFolderDataset(folder))
                            throw std::runtime_error(folder + " already has a " + std::string(DatasetManifest::kFileName) +
                                                     ": open it with Open.");
                        r.manifest = manifestOfOneStack(dir);
                        if (!r.manifest.files.empty()) {
                            try {
                                r.first = probeDataset(fromPath(dir / toPath(r.manifest.files.front().path)));
                                r.firstProbed = true;
                            } catch (const std::exception&) {
                                // unreadable first file: the open will say so properly
                            }
                        }
                    } catch (const std::exception& e) {
                        r.error = e.what();
                    }
                    post([this, alive, appPtr, r = std::move(r)] {
                        if (alive->load() && isOpen()) oneStackListed(*appPtr, r);
                    });
                });
            }

            void oneStackListed(App& app, const OneStackResult& r) {
                oneStackBusy_ = false;
                if (r.folder != path()) return;   // the path has changed since: this is another folder's
                if (!r.error.empty()) {
                    app.message("Open as one stack", r.error);
                    return;
                }
                if (r.manifest.files.empty()) {
                    app.message("Open as one stack", "No TIFF files in " + r.folder + ".", MessageIcon::Info);
                    return;
                }
                // A file's own pages become z; the files become t. That is right
                // for a time series and wrong for a stack saved a plane per file,
                // and the two look identical from the names, so say which reading
                // is about to be taken when the files are single planes.
                std::string caution;
                if (r.firstProbed && r.first.dims.z <= 1 && r.first.dims.t <= 1)
                    caution = format(" Each file holds a single plane, so this gives %d time points of one plane. If "
                                     "these are instead the planes of one stack, combine them into one TIFF first: a "
                                     "folder maps files to channels, tiles and time points, and takes z from the pages "
                                     "inside each file.",
                                     static_cast<int>(r.manifest.files.size()));
                const std::string text =
                    format("Read the %d files in %s as one stack, one time point each?", static_cast<int>(r.manifest.files.size()),
                           r.folder.c_str()) +
                    "\n\n" + "In name order, " + r.manifest.files.front().path + " first and " + r.manifest.files.back().path +
                    " last." + caution + " A " + DatasetManifest::kFileName +
                    " is written beside the files so the folder opens directly from then on; edit it, or use Folder…, for "
                    "channels, tiles or another order.";
                app.ask("Open as one stack", text, {"Cancel", "Open"},
                        [this, alive = alive_, appPtr = &app, manifest = r.manifest, folder = r.folder](int answer) {
                            if (answer != 1 || !alive->load()) return;
                            writeAndOpenOneStack(*appPtr, manifest, folder);
                        });
            }

            void writeAndOpenOneStack(App& app, const DatasetManifest& manifest, const std::string& folder) {
                try {
                    manifest.save(toPath(folder) / DatasetManifest::kFileName);
                    OpenOptions options;
                    options.tile = 0;
                    options.readAll = readAs_ == 1;
                    if (!app.bridge().openDatasetAsync(folder, options)) {
                        app.message("Open as one stack", "Another task is still running: cancel it or wait.");
                        return;
                    }
                } catch (const std::exception& e) {
                    app.message("Open as one stack", e.what());
                    return;
                }
                App::addRecentFile(folder);
                close();   // the dataset is open; the caller must not open it again
            }

            // --- drawing -------------------------------------------------------------------------

            // This computer | Cluster: where Browse looks. A cluster path is
            // "cluster://<host>/<path>", opened through the connected worker.
            void drawLocationRow(App& app) {
                if (isRemoteDatasetPath(path()) && location_ == 0 && !locationTouched_) location_ = 1;
                if (widgets::segmented("##where", {"This computer", "Cluster"}, &location_)) locationTouched_ = true;
                ImGui::SameLine(0.0f, px(10));
                if (location_ == 1) {
                    ClusterLink& link = app.cluster();
                    ImU32 color = theme::kNeutral600;
                    std::string state = link.indicator(color);
                    if (!link.sshUp()) {
                        widgets::text(state.empty() ? std::string("Not connected to a cluster") : state, 11, color);
                        ImGui::SameLine(0.0f, px(8));
                        if (widgets::linkButton("Connect to cluster\xE2\x80\xA6")) app.defer([&app] { app.clusterDialog(); });
                    } else {
                        widgets::text(link.connected() ? "Read on the cluster by the worker; only what is shown comes here"
                                                       : "Browsing works now; opening a dataset waits for the worker",
                                      11, theme::kNeutral600);
                    }
                }
            }

            void drawPathRow(App& app) {
                drawLocationRow(app);
                if (location_ == 1) {
                    drawClusterPathRow(app);
                    return;
                }
                const float spacing = px(6);
                const bool zarr = zarrSupported();
                const ImVec2 browse = buttonSize("Browse", widgets::ButtonKind::Secondary, true);
                const ImVec2 folder = buttonSize("Folder…", widgets::ButtonKind::Secondary, true);
                const ImVec2 directory = buttonSize("Directory…", widgets::ButtonKind::Secondary, true);
                const Line line;
                float fieldW = line.width() - browse.x - folder.x - 2 * spacing;
                if (zarr) fieldW -= directory.x + spacing;
                fieldW = std::max(px(80), fieldW);

                line.at(0.0f, line.height());
                widgets::FieldOpts f;
                f.width = fieldW / std::max(theme::scale(), 0.01f);
                f.hint = "/data/…/stack.tif, dataset.zarr or a folder of TIFF files";
                if (focusPath_) ImGui::SetKeyboardFocusHere();
                focusPath_ = false;
                if (widgets::inputText("##path", &path_, f)) pathChanged();

                widgets::ButtonOpts b;
                b.small = true;
                float x = fieldW + spacing;
                line.at(x, browse.y);
                if (widgets::button("Browse", b)) {
                    app.defer([this, alive = alive_, &app] {
                        if (!alive->load()) return;
                        const std::string start = path().empty() ? fastDirectory(app) : parentPath(path());
                        const std::string chosen = platform::openFileDialog("Open dataset", start, fileFilters());
                        if (!chosen.empty()) setPath(chosen);
                    });
                }
                x += browse.x + spacing;
                line.at(x, folder.y);
                b.tooltip = format("A folder of TIFF files, one per channel / time point / tile, described by a %s manifest",
                                   DatasetManifest::kFileName);
                if (widgets::button("Folder…", b)) {
                    app.defer([this, alive = alive_, &app] {
                        if (alive->load()) browseFolder(app);
                    });
                }
                if (zarr) {
                    x += folder.x + spacing;
                    line.at(x, directory.y);
                    b.tooltip = "zarr / N5 stores are directories";
                    if (widgets::button("Directory…", b)) {
                        app.defer([this, alive = alive_, &app] {
                            if (!alive->load()) return;
                            const std::string chosen = platform::pickFolderDialog("Open zarr / N5 store", fastDirectory(app));
                            if (!chosen.empty()) setPath(chosen);
                        });
                    }
                }
                line.end();
            }

            void drawClusterPathRow(App& app) {
                const float spacing = px(6);
                const ImVec2 browse = buttonSize("Browse", widgets::ButtonKind::Secondary, true);
                const Line line;
                const float fieldW = std::max(px(80), line.width() - browse.x - spacing);
                line.at(0.0f, line.height());
                widgets::FieldOpts f;
                f.width = fieldW / std::max(theme::scale(), 0.01f);
                f.hint = "cluster://fiona/home/\xE2\x80\xA6/stack.tif (Browse lists the cluster's folders)";
                if (widgets::inputText("##clusterPath", &path_, f)) pathChanged();
                widgets::ButtonOpts b;
                b.small = true;
                b.enabled = app.cluster().sshUp();
                b.tooltip = b.enabled ? std::string("The cluster's folders, through the SSH session")
                                      : std::string("Connect to the cluster first");
                line.at(fieldW + spacing, browse.y);
                if (widgets::button("Browse##cluster", b)) {
                    std::string start, host, remote;
                    if (splitClusterPath(path(), host, remote)) start = parentPathOf(remote);
                    app.showDialog(makeClusterBrowser(app, start, false, [this, alive = alive_](const std::string& chosen) {
                        if (alive->load()) setPath(chosen);
                    }));
                }
                line.end();
            }

            static std::string parentPathOf(const std::string& p) {
                const std::size_t slash = p.find_last_of('/');
                return slash == std::string::npos || slash == 0 ? std::string("/") : p.substr(0, slash);
            }

            // Where a file dialog starts when the path says nothing: the
            // directory of the last dataset, else the user's own.
            static std::string fastDirectory(App& app) {
                const std::string last = app.lastDir();
                return !last.empty() && isDirectory(last) ? last : platform::homeDirectory();
            }

            void browseFolder(App& app) {
                std::string start = path();
                if (!start.empty()) {
                    // Parent of a TIFF folder: listing the folder itself stats
                    // every stack on Vast/NFS and freezes the picker.
                    start = isDirectory(start) ? parentPath(start) : parentPath(parentPath(start));
                }
                if (start.empty() || !isDirectory(start)) start = fastDirectory(app);
                const std::string d = platform::pickFolderDialog("Open folder of TIFF files", start);
                if (!d.empty()) folderPicked(app, d);
            }

            void folderPicked(App& app, const std::string& d) {
                setPath(d);   // the probe reports the manifest, or its absence
                if (isFolderDataset(d)) return;
                // no manifest yet: describe the files; the folder dialog opens
                // the dataset itself, so this one then closes without asking
                // the caller to open it again
                app.setLastDir(d);
                app.showDialog(dataset_dialogs::makeFolderDatasetDialog(
                    app, d,
                    [this, alive = alive_, d] {
                        if (!alive->load()) return;
                        App::addRecentFile(d);
                        close();
                    },
                    readAs_ == 1));
            }

            void drawFacts() {
                gap(10);
                if (!facts_.empty()) widgets::textWrapped(facts_, 12, theme::kNeutral600);
                else if (error_.empty()) widgets::text(probing_ || probeAt_ ? "Reading…" : "", 12, theme::kNeutral600);
                if (!error_.empty()) {
                    if (!facts_.empty()) gap(6);
                    widgets::textWrapped(error_, 11, theme::kAccentText);
                }
            }

            void drawLayout() {
                const float scale = std::max(theme::scale(), 0.01f);
                const float spacing = px(10);
                gap(14);
                widgets::rule(2);
                gap(10);
                widgets::caption("Page layout");
                gap(8);
                const float column = std::floor((ImGui::GetContentRegionAvail().x - 3 * spacing) / 4.0f);
                widgets::FieldOpts f;
                f.width = column / scale;
                bool changed = false;

                Columns grid;
                grid.labelled(0.0f, "Order");
                std::vector<std::string> orders;
                for (const char* o : kOrders) orders.push_back(format("%s (%c fastest)", o, o[0]));
                widgets::combo("##order", &order_, orders, f);
                widgets::tooltip("Which axis changes fastest from page to page (ImageJ hyperstacks: c, then z, then t)");
                grid.track();

                // the Load step's ranges, or the file's own count when that is more
                grid.labelled(column + spacing, "Channels (c)");
                changed = widgets::inputInt("##c", &c_, 1, std::max<std::int64_t>(1024, probed_.dims.c), 1, f) || changed;
                grid.track();

                grid.labelled(2 * (column + spacing), "Time points (t)");
                changed = widgets::inputInt("##t", &t_, 1, std::max<std::int64_t>(1000000, probed_.dims.t), 1, f) || changed;
                grid.track();

                grid.labelled(3 * (column + spacing), "Planes (z)");
                const ImGuiID zId = ImGui::GetID("##z");
                const ImVec2 zMin = ImGui::GetCursorScreenPos();
                changed = widgets::inputInt("##z", &z_, 0, std::max<std::int64_t>(1000000, probed_.dims.z), 1, f) || changed;
                // the spin arrows are the last items: the field's rectangle again, for its tool tip
                const ImVec2 zMax(zMin.x + column, zMin.y + theme::snap(px(theme::kInputH)));
                ImGui::SetCursorScreenPos(zMin);
                ImGui::Dummy(ImVec2(zMax.x - zMin.x, zMax.y - zMin.y));
                widgets::tooltip("0 = derived from the page count");
                if (z_ == 0 && ImGui::GetActiveID() != zId) {
                    // the spin box's special value: 0 reads "auto" until it is edited
                    const float border = theme::crispPen(theme::kBorder);
                    const ImVec2 a(zMin.x + border, zMin.y + border);
                    const ImVec2 b(zMax.x - theme::snap(px(16)) - border, zMax.y - border);
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(a, b, theme::kBg);
                    widgets::drawTextIn(dl, ImVec2(a.x + px(8) - border, a.y), b, "auto", 13, theme::kText, theme::Weight::Regular, 0.0f,
                                        0.5f);
                }
                grid.track();
                grid.end();
                if (changed) updatePageCheck();

                if (!pageCheck_.empty()) {
                    gap(8);
                    widgets::text(pageCheck_, 11, pageCheckOk_ ? theme::kNeutral600 : theme::kAccentText);
                }

                gap(12);
                widgets::caption("Metadata");
                gap(8);
                const float third = std::floor((ImGui::GetContentRegionAvail().x - 2 * spacing) / 3.0f);
                Columns voxels;
                voxelFields(voxels, &voxel_[0], &voxel_[1], &voxel_[2], third, spacing);
                voxels.end();
                gap(8);
                widgets::fieldLabel("Channel names");
                widgets::FieldOpts names;
                names.hint = "488 α-actinin, 640 Mitochondria";
                widgets::inputText("##channels", &channels_, names);
                widgets::tooltip("Comma-separated channel names; a leading number is the emission wavelength");
                gap(10);
                simRow(sim_);
            }

            void drawReadAs() {
                gap(14);
                const Line line;
                const std::string label = "Read as";
                const float labelW = theme::textSize(label, 11).x + px(8);
                line.text(0.0f, label, 11, theme::kNeutral700);
                line.at(labelW, line.height());
                widgets::FieldOpts f;
                f.width = (line.width() - labelW) / std::max(theme::scale(), 0.01f);
                widgets::combo("##readAs", &readAs_, {"Lazy (planes on demand)", "Full load to RAM"}, f);
                line.end();
            }

            void drawRecent() {
                gap(14);
                widgets::rule(2);
                gap(10);
                widgets::caption("Recent datasets");
                gap(8);
                const float rowH = theme::snap(px(24));
                const float headerH = theme::snap(px(24));
                const float rows = static_cast<float>(std::max<std::size_t>(recent_.size(), 2));
                const float height = std::min(px(150), headerH + rows * rowH + px(4));
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, px(8, 0));
                ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, theme::kDivider);
                ImGui::PushStyleColor(ImGuiCol_Header, theme::kSurface);
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
                const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_NoSavedSettings |
                                              ImGuiTableFlags_SizingFixedFit;
                int opened = -1;
                if (ImGui::BeginTable("##recent", 3, flags, ImVec2(0.0f, height))) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("NAME", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("FORMAT", ImGuiTableColumnFlags_WidthFixed, px(80));
                    ImGui::TableSetupColumn("MODIFIED", ImGuiTableColumnFlags_WidthFixed, px(110));
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, headerH);
                    const char* heads[3] = {"NAME", "FORMAT", "MODIFIED"};
                    for (int c = 0; c < 3; ++c) {
                        ImGui::TableSetColumnIndex(c);
                        cellText(heads[c], headerH, theme::kCaptionPx, theme::kNeutral600, true);
                    }
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, theme::kSurface);
                    for (std::size_t i = 0; i < recent_.size(); ++i) {
                        const RecentRow& row = recent_[i];
                        ImGui::PushID(static_cast<int>(i));
                        ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                        ImGui::TableSetColumnIndex(0);
                        const bool selected = selectedRecent_ == static_cast<int>(i);
                        if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick,
                                              ImVec2(0.0f, rowH))) {
                            selectedRecent_ = static_cast<int>(i);
                            setPath(row.path);
                            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) opened = static_cast<int>(i);
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        widgets::tooltip(row.path);
                        ImGui::SameLine(0.0f, 0.0f);
                        const bool missing = row.modified == "missing";
                        cellText(widgets::elideText(row.name, ImGui::GetContentRegionAvail().x, 13), rowH, 13,
                                 missing ? theme::kNeutral500 : theme::kText, false);
                        ImGui::TableSetColumnIndex(1);
                        cellText(row.format, rowH, 13, missing ? theme::kNeutral500 : theme::kText, false);
                        ImGui::TableSetColumnIndex(2);
                        cellText(row.modified, rowH, 13, missing ? theme::kNeutral500 : theme::kText, false);
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                ImGui::PopStyleColor(4);
                ImGui::PopStyleVar();
                if (opened >= 0) accept();
            }

            // Text centred on a table row of `height` display pixels.
            static void cellText(const std::string& s, float height, float fontPx, ImU32 color, bool isCaption) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                ImGui::Dummy(ImVec2(0.0f, height));
                if (s.empty()) return;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (isCaption) {
                    const std::string& t = s;   // the header as written
                    const theme::FontScope f(fontPx, theme::captionFont());
                    const ImVec2 ts = ImGui::CalcTextSize(t.c_str());
                    dl->AddText(ImVec2(theme::snap(p.x), theme::snap(p.y + (height - ts.y) * 0.5f)), color, t.c_str());
                    return;
                }
                const ImVec2 ts = theme::textSize(s, fontPx);
                widgets::drawText(dl, ImVec2(p.x, p.y + (height - ts.y) * 0.5f), s, fontPx, color);
            }

            void drawButtons(App& app) {
                gap(14);
                std::vector<FooterButton> buttons;
                buttons.push_back({"Cancel", widgets::ButtonKind::Ghost, true, {}});
                if (oneStackVisible_)
                    buttons.push_back({"Open as one stack", widgets::ButtonKind::Secondary, !oneStackBusy_,
                                       "Every TIFF in the folder as one time point of one stack, in name order"});
                buttons.push_back({"Open", widgets::ButtonKind::Primary, openEnabled_, {}});
                const int pressed = footer(buttons);
                const int last = static_cast<int>(buttons.size()) - 1;
                if (pressed == 0) cancel();
                else if (pressed == last) accept();
                else if (pressed == 1) openAsOneStack(app);
                else if (enterPressed(popupAtStart_)) accept();
            }

            Accepted accepted_;

            std::string path_;
            bool focusPath_ = true;
            int location_ = 0;              // 0 this computer, 1 the cluster
            bool locationTouched_ = false;
            bool popupAtStart_ = false;
            std::string facts_;
            std::string error_;
            bool showLayout_ = true;
            int order_ = 0;
            std::int64_t c_ = 1, t_ = 1, z_ = 0;
            std::string pageCheck_;
            bool pageCheckOk_ = true;
            // typical widefield sampling until a file says otherwise
            std::array<double, 3> voxel_{0.1, 0.1, 0.2};
            std::string channels_;
            std::string probedChannels_;     // channels_ as the probe filled it
            SimFields sim_;
            int readAs_ = 1;
            std::vector<RecentRow> recent_;
            int selectedRecent_ = -1;
            bool openEnabled_ = false;
            bool oneStackVisible_ = false;   // a folder of TIFFs, a frame per file
            bool oneStackBusy_ = false;

            std::optional<Clock::time_point> probeAt_;
            bool probing_ = false;
            std::uint64_t generation_ = 0;
            // generation_ for the thread: a queued probe it has overtaken is skipped
            std::shared_ptr<std::atomic<std::uint64_t>> latestProbe_ = std::make_shared<std::atomic<std::uint64_t>>(0);
            bool acceptPending_ = false;
            DatasetMeta probed_;
            bool probeOk_ = false;
            bool dimsFromMetadata_ = false;
            bool isFolder_ = false;          // a folder with a manifest: nothing to override
            Index pages_ = 0;

            bool acceptedNow_ = false;
            std::string acceptedPath_;
            OpenOptions acceptedOptions_;

            Alive alive_ = makeAlive();
            Worker worker_;                  // the probe and the one-stack listing
            Worker scanWorker_;              // the recent files' dates, which may wait on the network
        };

    } // namespace

    std::shared_ptr<Dialog> makeOpenDatasetDialog(App& app, const std::string& initialPath,
                                                  std::function<void(const std::string&, const OpenOptions&)> accepted) {
        return std::make_shared<OpenDatasetDialog>(app, initialPath, std::move(accepted));
    }

} // namespace sirius::app::gui
