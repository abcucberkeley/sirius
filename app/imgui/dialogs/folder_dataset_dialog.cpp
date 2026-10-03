// "Open folder as dataset": a folder of TIFF stacks described by a filename
// pattern with named groups, previewed live, then opened. The pattern is
// remembered (last used, and per folder) so the next acquisition of the same
// layout does not need a new regex. An existing manifest can be loaded to
// recover its pattern; a sidecar is written only when the mapping is new or
// the TIFF folder cannot hold one (then a local cache file with files_folder).
//
// A folder of an acquisition holds thousands of files, often on a network
// drive, and std::regex is not quick: the directory is listed, the pattern
// matched and the manifest built (one TIFF header per tile) on the dialog's
// own thread. Matching waits 150 ms after the last key, and a result is
// taken only when the pattern is still the one it was matched for.
//
// The same dialog serves a folder on the cluster (File ▸ Open folder as
// dataset with a cluster session, the Load step's Source in Folder mode on
// the cluster): the names come over the session's command channel, the
// pattern and preview are the same, the stacks' shapes are asked of the
// engine on the node, and the manifest is kept in ~/.sirius/manifests on the
// cluster -- never in the data folder (core/cluster_folder.hpp). The Load
// step's Source is then that manifest's cluster:// path.
#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>
#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/cluster_folder.hpp"
#include "core/host.hpp"
#include "core/manifest.hpp"
#include "core/remote_host.hpp"
#include "core/remote_source.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/open_dataset_common.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {
        using namespace dataset_dialogs;
        using theme::px;
        namespace fs = std::filesystem;
        using Clock = std::chrono::steady_clock;

        constexpr int kPreviewDelayMs = 150;
        constexpr const char* kTitle = "Open folder as dataset";

        // Preset patterns for the layouts acquisition software writes most
        // often. The button keeps the chosen name so the pick is visible.
        struct Preset {
            const char* label;
            const char* pattern;
            FilenameRule::Positions positions;
            const char* example;   // tooltip; kept off the list so it stays short
        };

        const Preset kPresets[] = {
            // ABC AOLLS / LabVIEW SPIM (Aang_Foundation, Scan_Iter_*_CamA_*_000x_000y_000z_0000t.tif)
            {"AOLLS",
             R"(Scan_Iter_\d+_\d+_\d+_\d+_(?P<channel>Cam[A-Z])_ch\d+_CAM\d+_stack\d+_\d+nm_\d+msec_\d+msecAbs_(?P<x>\d+)x_(?P<y>\d+)y_(?P<z>\d+)z_(?P<t>\d+)t\.tiff?$)",
             FilenameRule::Positions::GridIndex, "Scan_Iter_…_CamA_ch0_…_000x_000y_000z_0000t.tif"},
            {"channel · t · x · y", R"(^.*_c(?P<channel>[^_]+)_t(?P<t>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.tiff?$)",
             FilenameRule::Positions::GridIndex, "stack_c488_t0_x1_y2.tif"},
            {"channel · t", R"(^.*_ch(?P<channel>\d+)_t(?P<t>\d+)\.tiff?$)", FilenameRule::Positions::None, "stack_ch0_t003.tif"},
            {"channel prefix", R"(^(?P<channel>[^_]+)_.*\.tiff?$)", FilenameRule::Positions::None, "488_cell.tif"},
            {"Micro-Manager positions", R"(^.*_MMStack_Pos(?P<tile>\d+)\.ome\.tiff?$)", FilenameRule::Positions::None,
             "run_MMStack_Pos3.ome.tif"},
            {"stage coordinates", R"(^.*_c(?P<channel>[^_]+)_X(?P<x>-?[\d.]+)_Y(?P<y>-?[\d.]+)\.tiff?$)",
             FilenameRule::Positions::Microns, "stack_c488_X1200.5_Y-30.0.tif"},
        };

        bool looksLikeAollsScanIter(const std::string& name) {
            return name.size() > 10 && name.compare(0, 10, "Scan_Iter_") == 0 && name.find("msecAbs_") != std::string::npos &&
                   name.find("Cam") != std::string::npos;
        }

        // 1-based; 0 when the pattern is none of the presets.
        int indexOfPresetPattern(const std::string& pat) {
            for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i)
                if (pat == kPresets[i].pattern) return i + 1;
            return 0;
        }

        std::string presetLabelFor(const std::string& pat) {
            const int i = indexOfPresetPattern(pat);
            return i > 0 ? std::string(kPresets[i - 1].label) : std::string("Presets");
        }

        int positionsIndex(FilenameRule::Positions p) {
            return p == FilenameRule::Positions::GridIndex ? 1 : p == FilenameRule::Positions::Microns ? 2
                                                                                                       : 0;
        }

        // positionsIndex for a manifest that does not say how its tiles were
        // placed (written before it recorded that, or by hand), from the tiles
        // themselves. manifestFromFolder gives every tile a grid index, so the
        // indices alone say nothing: none when every tile sits at the origin;
        // grid indices when each axis's position is its index times one step;
        // else stage coordinates, whose indices are their ranks. Evenly spaced
        // coordinates from 0 look like a grid: the recorded mode is what settles it.
        int guessedPositionsIndex(const DatasetManifest& m) {
            const bool anyPosition = std::any_of(m.tiles.begin(), m.tiles.end(), [](const TileInfo& t) {
                return t.positionUm[0] != 0.0 || t.positionUm[1] != 0.0 || t.positionUm[2] != 0.0;
            });
            if (m.tiles.size() <= 1 || !anyPosition) return 0;
            for (std::size_t k = 0; k < 3; ++k) {
                std::optional<double> step;
                for (const TileInfo& t : m.tiles) {
                    const double p = t.positionUm[k];
                    if (t.gridIndex[k] == 0) {
                        if (std::abs(p) > 1e-9) return 2;
                        continue;
                    }
                    const double s = p / static_cast<double>(t.gridIndex[k]);
                    if (!step) step = s;
                    else if (std::abs(s - *step) > 1e-6 * std::max(1.0, std::abs(*step))) return 2;
                }
            }
            return 1;
        }

        // First of the group aliases that matched, else empty.
        const std::string& group(const FilenameMatch& m, std::initializer_list<const char*> names) {
            static const std::string none;
            for (const char* n : names) {
                const auto it = m.groups.find(n);
                if (it != m.groups.end()) return it->second;
            }
            return none;
        }

        // The whole of the trimmed `s` as a number, nothing left over.
        bool toNumber(const std::string& s, double& out) {
            const std::string t = trimmed(s);
            if (t.empty()) return false;
            char* end = nullptr;
            const double v = std::strtod(t.c_str(), &end);
            if (!end || end == t.c_str() || *end != '\0') return false;
            out = v;
            return true;
        }

        // "488" < "561" < "640" numerically, everything else by name.
        bool tokenLess(const std::string& a, const std::string& b) {
            double da = 0.0, db = 0.0;
            const bool na = toNumber(a, da), nb = toNumber(b, db);
            if (na && nb) return da < db;
            if (na != nb) return na;
            return a < b;
        }

        std::string tileKey(const FilenameMatch& m) {
            return group(m, {"tile"}) + "|" + group(m, {"x", "col"}) + "|" + group(m, {"y", "row"}) + "|" + group(m, {"z"});
        }

        bool samePath(const fs::path& a, const fs::path& b) { return canonicalPath(a) == canonicalPath(b); }

        constexpr const char* kLastManifestDirKey = "folderDataset/lastManifestDir";
        // where a new sidecar goes when nothing was chosen: "data" (beside the
        // TIFFs), "cache" (with the application's files), unset = ask
        constexpr const char* kSidecarPlaceKey = "folderDataset/sidecarPlace";
        constexpr const char* kLastPatternKey = "folderDataset/lastPattern";
        constexpr const char* kLastPositionsKey = "folderDataset/lastPositions";
        constexpr const char* kLastOverlapKey = "folderDataset/lastOverlap";
        constexpr const char* kPatternMapKey = "folderDataset/patternMap";

        // Per-folder regex if this directory was opened before, else the last
        // pattern that successfully opened a folder (most acquisitions share one).
        std::string rememberedPatternFor(const std::string& canonicalFolder) {
            const nlohmann::json map = settings().value(kPatternMapKey);
            if (map.is_object()) {
                const auto it = map.find(canonicalFolder);
                if (it != map.end() && it->is_string() && !it->get<std::string>().empty()) return it->get<std::string>();
            }
            return settings().getString(kLastPatternKey);
        }

        void rememberPatternFor(const std::string& canonicalFolder, const std::string& pattern, int positions, double overlap) {
            const std::string pat = trimmed(pattern);
            if (pat.empty()) return;
            settings().set(kLastPatternKey, pat);
            settings().set(kLastPositionsKey, positions);
            settings().set(kLastOverlapKey, overlap);
            nlohmann::json map = settings().value(kPatternMapKey);
            if (!map.is_object()) map = nlohmann::json::object();
            map[canonicalFolder] = pat;
            settings().set(kPatternMapKey, map);
        }

        // The sidecar of a folder that cannot (or shall not) hold one: with the
        // application's own files, named after the folder's path.
        fs::path cachedManifestPath(const std::string& canonicalFolder) {
            std::uint64_t h = 1469598103934665603ull;   // FNV-1a
            for (const char c : canonicalFolder) {
                h ^= static_cast<unsigned char>(c);
                h *= 1099511628211ull;
            }
            const std::string dir = absolutePath(settings().directory()) + "/folder-manifests";
            platform::makePath(dir);
            return toPath(dir) / (format("%016llx", static_cast<unsigned long long>(h)) + ".toml");
        }

        // A small place to start a file dialog in: the cluster's scratch the
        // home directory maps onto when there is one, else home.
        std::string fastWritableDirectory() {
            const std::string home = platform::homeDirectory();
            const std::string nvme = "/clusterfs/nvme2/Users/" + fileName(home);
            if (!fileName(home).empty() && isDirectory(nvme)) return nvme;
            return home;
        }

        // Outside the TIFF folder, so the dialog does not list hundreds of stacks.
        std::string manifestBrowseDirectory(const std::string& dest, const std::string& tiffFolder) {
            const fs::path tiff = toPath(absolutePath(tiffFolder)).lexically_normal();
            auto usable = [&tiff](const std::string& dir) {
                return !dir.empty() && toPath(absolutePath(dir)).lexically_normal() != tiff && isDirectory(dir);
            };
            const std::string destDir = dest.empty() ? std::string() : parentPath(dest);
            if (usable(destDir)) return destDir;
            const std::string last = settings().getString(kLastManifestDirKey);
            if (usable(last)) return absolutePath(last);
            return fastWritableDirectory();
        }

        std::vector<platform::FileFilter> manifestFilters() { return {{"SIRIUS dataset", "toml"}, {"All files", "*"}}; }

        // --- what the thread reports ---------------------------------------------------

        using Names = std::shared_ptr<const std::vector<std::string>>;

        struct Listing {
            std::uint64_t generation = 0;               // of the folder it lists (the folder can change)
            Names names;                                // the folder's TIFF files, sorted
            std::optional<DatasetManifest> existing;    // a manifest already in the folder
            std::string canonicalFolder;
            std::shared_ptr<const ClusterFolder> cluster;   // a folder on the cluster: as it was listed
            std::string error;                          // the cluster's folder could not be listed
        };

        struct MatchResult {
            std::uint64_t generation = 0;
            std::vector<FilenameMatch> matches;
            std::string error;
        };

        // What the dialog remembers for a folder once it has opened
        // (rememberPatternFor), as the fields were when Open was pressed.
        struct RememberedFields {
            std::string pattern;
            int positions = 1;
            double overlap = 10.0;
        };

        struct BuildRequest {
            std::string folder;
            std::string destText;
            FilenameRule rule;
            RememberedFields fields;
            std::string loadedManifestPath;
            std::optional<DatasetManifest> existing;
        };

        struct BuildResult {
            std::string openPath;        // a folder on the cluster: its manifest's cluster:// path, written
            fs::path dest;
            bool openAsIs = false;       // the loaded manifest is valid: nothing to write
            DatasetManifest manifest;
            RememberedFields fields;     // the request's
            std::string error;
            bool destExists = false;
            bool destInFolder = false;
        };

        BuildResult build(const BuildRequest& q) {
            BuildResult r;
            r.fields = q.fields;
            try {
                const fs::path folderPath = toPath(q.folder);
                r.dest = toPath(q.destText);
                std::error_code ec;
                if (fs::is_directory(r.dest, ec)) r.dest /= DatasetManifest::kFileName;
                if (r.dest.extension().empty()) r.dest += ".toml";

                const bool destIsLoaded = !q.loadedManifestPath.empty() && samePath(r.dest, toPath(q.loadedManifestPath));
                if (destIsLoaded && q.existing && q.existing->pattern == q.rule.pattern &&
                    samePath(q.existing->filesRoot(r.dest), folderPath)) {
                    if (q.existing->validate(folderPath).empty()) {
                        r.openAsIs = true;
                        return r;
                    }
                }
                try {
                    r.manifest = manifestFromFolder(folderPath, q.rule, nullptr);
                } catch (const std::exception& e) {
                    r.error = std::string("The files do not form a dataset:\n") + e.what();
                    return r;
                }
                r.destExists = fs::exists(r.dest, ec);
                r.destInFolder = samePath(r.dest.parent_path(), folderPath);
            } catch (const std::exception& e) {
                r.error = e.what();
            }
            return r;
        }

        // --- the question where the sidecar goes ------------------------------------------

        // A message box with a "Do the same for other folders" check box, which
        // App::ask has not.
        class SidecarQuestion final : public Dialog {
        public:
            enum Answer { Cancelled,
                          Cache,
                          Data };
            using Answered = std::function<void(Answer, bool remember)>;

            SidecarQuestion(std::string folder, Answered answered) : folder_(std::move(folder)), answered_(std::move(answered)) {}

            std::string title() const override { return kTitle; }
            ImVec2 size() const override { return ImVec2(580, 0); }

            void draw(App&) override {
                const bool popupAtStart = popupAbove();
                widgets::textWrapped(format("Write %s into the data folder?", DatasetManifest::kFileName), 13, theme::kText,
                                     theme::Weight::SemiBold);
                gap(10);
                widgets::textWrapped(folder_ +
                                         "\n\nBeside the files, the folder opens directly from then on, for everyone who uses it. "
                                         "Kept with this application's own files instead, nothing is added to the data folder, and "
                                         "this dialog remembers the pattern for it.",
                                     13, theme::kNeutral800);
                gap(12);
                widgets::checkbox("Do the same for other folders", &remember_);
                gap(14);
                const int pressed = footer({{"Cancel", widgets::ButtonKind::Ghost, true, {}},
                                            {"Write into the data folder", widgets::ButtonKind::Secondary, true, {}},
                                            {"Keep it with the application", widgets::ButtonKind::Primary, true, {}}});
                if (pressed == 1) answer_ = Data;
                else if (pressed == 2 || (pressed < 0 && enterPressed(popupAtStart))) answer_ = Cache;
                if (pressed >= 0 || answer_ != Cancelled) close();
            }

            void closed(App&) override {
                if (answered_) answered_(answer_, remember_);
            }

        private:
            std::string folder_;
            Answered answered_;
            Answer answer_ = Cancelled;
            bool remember_ = false;
        };

        // --- the dialog -------------------------------------------------------------------

        class FolderDatasetDialog final : public Dialog {
        public:
            // `folder`: a folder on this computer, a cluster folder
            // ("cluster://<host>/<path>"), or "" to choose one here (on the
            // cluster when a session is up and the HPC backend computes there).
            FolderDatasetDialog(App& app, const std::string& folder, std::function<void()> opened, bool readAll)
                : app_(app), bridge_(app.bridge()), opened_(std::move(opened)), readAll_(readAll), worker_(app.bridge()) {
                std::string host, remotePath;
                // screenshots only: the folder File ▸ Open folder as dataset would have been given
                const std::string given = folder.empty() ? host::environment("SIRIUS_TEST_FOLDER_DATASET") : folder;
                if (splitClusterPath(given, host, remotePath)) {
                    remote_ = true;
                    host_ = host;
                    folder_ = remotePath;
                } else if (!given.empty()) {
                    folder_ = absolutePath(given);
                } else {
                    remote_ = app.cluster().sshUp() && (app.cluster().connected() || app.wb().backend() == Backend::Hpc);
                    if (remote_) host_ = app.cluster().status().host;
                }
                startListing();
            }

            ~FolderDatasetDialog() override { alive_->store(false); }

            std::string title() const override { return kTitle; }
            // tall enough for the layout's minimum, about 880
            ImVec2 size() const override { return ImVec2(720, 900); }
            bool resizable() const override { return true; }

            void draw(App& app) override {
                popupAtStart_ = popupAbove();
                cellEditing_ = false;
                if (previewAt_) {
                    if (Clock::now() >= *previewAt_) runPreview();
                    else app.requestRedraw();   // the timer: frames until it is due
                }
                // While the manifest is being built, the fields are what it
                // is built from: an edit then would be lost, or remembered for
                // the folder without having opened it.
                const bool building = building_;
                drawFolderRows(app);
                ImGui::BeginDisabled(building);
                drawPattern();
                ImGui::EndDisabled();
                drawPreview();
                const float lowerTop = ImGui::GetCursorPosY();
                ImGui::BeginDisabled(building);
                drawPositions();
                drawMetadata();
                drawChannelsAndTiles();
                ImGui::EndDisabled();
                drawButtons();
                const float lower = ImGui::GetCursorPosY() - lowerTop;
                if (std::abs(lower - lowerHeight_) > 0.5f) {
                    lowerHeight_ = lower;
                    app.requestRedraw();   // the preview takes what is left: lay out again
                }
            }

        private:
            struct Chan {
                ChannelInfo info;
                bool customColor = false;   // chosen here or preloaded: not re-derived from the wavelength
                std::string labelText;      // as typed
                std::string nmText;
            };

            FilenameRule::Positions positionsMode() const {
                switch (positions_) {
                    case 1: return FilenameRule::Positions::GridIndex;
                    case 2: return FilenameRule::Positions::Microns;
                    default: return FilenameRule::Positions::None;
                }
            }

            FilenameRule rule() const {
                FilenameRule r;
                r.pattern = trimmed(pattern_);
                r.positions = positionsMode();
                r.overlapFraction = overlap_ / 100.0;
                r.voxelUm = voxel_;
                r.frameIntervalS = interval_;
                r.sim.present = sim_.present;
                r.sim.ndirs = static_cast<int>(sim_.dirs);
                r.sim.nphases = static_cast<int>(sim_.phases);
                r.sim.fastSi = sim_.fastSi;
                r.acquisition = trimmed(acquisition_);
                for (const std::string& token : tokens_) {
                    const auto it = chans_.find(token);
                    if (it != chans_.end()) r.channelInfo[token] = it->second.info;
                }
                return r;
            }

            // --- loading -----------------------------------------------------------------

            // The folder is listed: preload, then match. In-folder manifest, else
            // the regex remembered for this folder, else the last pattern that
            // opened any folder, else AOLLS when the names look like it.
            // The folder (again): on this computer read here, on the cluster
            // over the session's command channel. What the last folder showed
            // goes; the pattern and the metadata typed stay.
            void startListing() {
                const std::uint64_t generation = ++listGeneration_;
                names_.reset();
                cluster_.reset();
                existing_.reset();
                loadedManifestPath_.clear();
                matches_.clear();
                matchedCount_ = 0;
                canOpen_ = false;
                statusBad_ = false;
                ++generation_;   // a match under way is for the last folder
                matching_ = false;
                manifestPath_ = remote_ || folder_.empty() ? std::string() : folder_ + "/" + DatasetManifest::kFileName;
                if (folder_.empty()) {
                    status_ = remote_ ? "Choose a folder on the cluster: Browse." : "Choose a folder: Browse.";
                    return;
                }
                if (remote_) {
                    if (!app_.cluster().sshUp()) {
                        status_ = "Not logged in to the cluster: connect first (Cluster button).";
                        statusBad_ = true;
                        return;
                    }
                    status_ = "Listing the folder on " + host_ + "…";
                    cluster::Session* session = &app_.cluster().session();
                    worker_.run([this, alive = alive_, session, host = host_, dir = folder_, generation](const Worker::Post& post) {
                        Listing l;
                        l.generation = generation;
                        try {
                            auto f = std::make_shared<ClusterFolder>(listClusterFolder(*session, host, dir));
                            l.canonicalFolder = f->clusterPath();
                            l.names = std::make_shared<const std::vector<std::string>>(f->tiffs);
                            l.existing = f->existing;
                            l.cluster = std::move(f);
                        } catch (const std::exception& e) {
                            l.error = e.what();
                            l.names = std::make_shared<const std::vector<std::string>>();
                            l.canonicalFolder = makeClusterPath(host, dir);
                        }
                        post([this, alive, l = std::move(l)] {
                            if (alive->load() && isOpen()) listed(l);
                        });
                    });
                    return;
                }
                status_ = "Matching files…";
                worker_.run([this, alive = alive_, dir = folder_, generation](const Worker::Post& post) {
                    Listing l;
                    l.generation = generation;
                    std::vector<std::string> names;
                    try {
                        const fs::path path = toPath(dir);
                        l.canonicalFolder = fromPath(canonicalPath(path));
                        names = tiffNamesInOrder(path);
                        std::error_code ec;
                        const fs::path manifest = path / DatasetManifest::kFileName;
                        if (fs::exists(manifest, ec)) {
                            try {
                                l.existing = DatasetManifest::load(manifest);
                            } catch (const std::exception&) {
                                l.existing.reset();
                            }
                        }
                    } catch (const std::exception&) {
                        // an unreadable folder holds no files, which the status line says
                    }
                    if (l.canonicalFolder.empty()) l.canonicalFolder = dir;
                    l.names = std::make_shared<const std::vector<std::string>>(std::move(names));
                    post([this, alive, l = std::move(l)] {
                        if (alive->load() && isOpen()) listed(l);
                    });
                });
            }

            // A folder chosen here: "cluster://<host>/<path>" or one of this computer.
            void setFolder(const std::string& folder) {
                std::string host, remotePath;
                if (splitClusterPath(folder, host, remotePath)) {
                    remote_ = true;
                    host_ = host;
                    folder_ = remotePath;
                } else {
                    remote_ = false;
                    folder_ = folder.empty() ? std::string() : absolutePath(folder);
                }
                startListing();
            }

            // This computer | Cluster switched: the folder of the other side is chosen anew.
            void setWhere(bool remote) {
                if (remote == remote_) return;
                remote_ = remote;
                host_ = remote ? app_.cluster().status().host : std::string();
                folder_.clear();
                startListing();
            }

            void browse(App& app) {
                if (remote_) {
                    if (!app.cluster().sshUp()) {
                        app.clusterDialog();
                        return;
                    }
                    app.showDialog(makeClusterBrowser(app, folder_, true, [this, alive = alive_](const std::string& chosen) {
                        if (alive->load() && isOpen() && !chosen.empty()) setFolder(chosen);
                    }));
                    return;
                }
                std::string start = folder_.empty() ? app.lastDir() : parentPath(folder_);
                if (start.empty() || !isDirectory(start)) start = platform::homeDirectory();
                const std::string chosen = platform::pickFolderDialog("Open folder as dataset", start);
                if (chosen.empty()) return;
                app.setLastDir(chosen);
                setFolder(chosen);
            }

            void listed(const Listing& l) {
                if (l.generation != listGeneration_) return;   // a folder chosen since
                if (!l.error.empty()) {
                    names_ = l.names;
                    canonicalFolder_ = l.canonicalFolder;
                    status_ = "Could not list the folder on " + host_ + ": " + l.error;
                    statusBad_ = true;
                    canOpen_ = false;
                    return;
                }
                cluster_ = l.cluster;
                names_ = l.names;
                canonicalFolder_ = l.canonicalFolder;
                // a manifest loaded (Load…) while the folder was being listed stands
                if (!existing_) existing_ = l.existing;
                const std::string remembered = rememberedPatternFor(canonicalFolder_);
                if (touched_) {
                    // typed while the folder was being listed: what was typed stands
                } else if (existing_) {
                    if (!remote_) loadedManifestPath_ = folder_ + "/" + DatasetManifest::kFileName;
                    applyManifest(*existing_);
                    if (trimmed(pattern_).empty() && !remembered.empty()) setPattern(remembered);
                } else if (!remembered.empty()) {
                    setPattern(remembered);
                    positions_ = std::clamp(settings().getInt(kLastPositionsKey, 1), 0, 2);
                    overlap_ = std::clamp(settings().getDouble(kLastOverlapKey, 10.0), 0.0, 90.0);
                } else if (std::any_of(names_->begin(), names_->end(), looksLikeAollsScanIter)) {
                    setPattern(kPresets[0].pattern);
                    positions_ = 1;
                }
                runPreview();
            }

            void setPattern(const std::string& p) {
                pattern_ = p;
                presetLabel_ = presetLabelFor(p);
            }

            void applyManifest(const DatasetManifest& m) {
                voxel_ = {m.voxelUm[0] > 0.0 ? m.voxelUm[0] : 0.1, m.voxelUm[1] > 0.0 ? m.voxelUm[1] : 0.1,
                          m.voxelUm[2] > 0.0 ? m.voxelUm[2] : 0.2};
                interval_ = std::clamp(m.frameIntervalS, 0.0, 1.0e6);
                acquisition_ = m.acquisition;
                sim_.present = m.sim.present;
                sim_.dirs = std::clamp(m.sim.ndirs, 1, 9);
                sim_.phases = std::clamp(m.sim.nphases, 1, 15);
                sim_.fastSi = m.sim.fastSi;
                // how the rule placed the tiles, when the manifest recorded it
                if (const std::optional<FilenameRule::Positions> p = positionsFromName(m.positions)) positions_ = positionsIndex(*p);
                else positions_ = guessedPositionsIndex(m);
                if (m.overlapFraction && std::isfinite(*m.overlapFraction))
                    overlap_ = std::clamp(*m.overlapFraction * 100.0, 0.0, 90.0);
                if (!m.pattern.empty()) setPattern(m.pattern);
                existing_ = m;
            }

            void loadManifestFile(App& app) {
                const std::string dest = trimmed(manifestPath_);
                std::string start = dest;
                if (start.empty() || isDirectory(start)) start = settings().getString(kLastManifestDirKey);
                if (start.empty() || !pathExists(start)) start = fastWritableDirectory();
                const std::string chosen = platform::openFileDialog("Load dataset manifest", start, manifestFilters());
                if (chosen.empty()) return;
                DatasetManifest m;
                try {
                    m = DatasetManifest::load(toPath(chosen));
                } catch (const std::exception& e) {
                    app.message("Load manifest", "Could not read " + chosen + ":\n" + e.what());
                    return;
                }
                touched_ = true;
                applyManifest(m);
                // on the cluster a manifest of this computer is a template: its pattern and metadata
                if (!remote_ && samePath(m.filesRoot(toPath(chosen)), toPath(folder_))) {
                    manifestPath_ = chosen;
                    loadedManifestPath_ = chosen;
                }
                settings().set(kLastManifestDirKey, parentPath(chosen));
                runPreview();
            }

            void saveManifestAs() {
                const std::string dest = trimmed(manifestPath_);
                std::string name = fileName(dest);
                if (name.empty()) name = DatasetManifest::kFileName;
                const std::string dir = manifestBrowseDirectory(dest, folder_);
                if (!dest.empty() &&
                    toPath(parentPath(dest)).lexically_normal() == toPath(absolutePath(folder_)).lexically_normal()) {
                    const std::string folderName = fileName(folder_);
                    if (!folderName.empty()) name = folderName + ".toml";
                }
                std::string chosen = platform::saveFileDialog("Save dataset manifest", dir, name, manifestFilters());
                if (chosen.empty()) return;
                if (!endsWithNoCase(chosen, ".toml")) chosen += ".toml";
                manifestPath_ = chosen;
                settings().set(kLastManifestDirKey, parentPath(chosen));
            }

            // --- the preview ---------------------------------------------------------------

            void patternEdited() {
                touched_ = true;
                previewAt_ = Clock::now() + std::chrono::milliseconds(kPreviewDelayMs);
                presetLabel_ = presetLabelFor(pattern_);
            }

            void applyPreset(int i) {
                if (i < 0 || i >= static_cast<int>(std::size(kPresets))) return;
                const Preset& p = kPresets[i];
                touched_ = true;
                pattern_ = p.pattern;
                positions_ = positionsIndex(p.positions);
                presetLabel_ = p.label;
                runPreview();
            }

            // Match the pattern against the folder; applyMatches fills the
            // preview and derives the channel table and tile map from what matched.
            void runPreview() {
                previewAt_.reset();
                if (!names_) return;   // still listing; listed() comes back here
                const std::uint64_t generation = ++generation_;
                const std::string pat = trimmed(pattern_);
                if (pat.empty()) {
                    MatchResult r;
                    r.generation = generation;
                    applyMatches(r, pat);
                    return;
                }
                matching_ = true;
                worker_.run([this, alive = alive_, names = names_, pat, generation](const Worker::Post& post) {
                    if (!alive->load()) return;
                    MatchResult r;
                    r.generation = generation;
                    try {
                        r.matches = matchFilenames(*names, pat);
                    } catch (const std::exception& e) {
                        r.error = e.what();
                        r.matches.clear();
                    }
                    post([this, alive, pat, r = std::move(r)]() mutable {
                        if (alive->load() && isOpen()) applyMatches(r, pat);
                    });
                });
            }

            void applyMatches(MatchResult& r, const std::string& pat) {
                if (r.generation != generation_) return;   // the pattern has changed since
                matching_ = false;
                patternOk_ = !pat.empty() && r.error.empty();
                matches_ = std::move(r.matches);
                matchedCount_ = 0;
                if (matches_.empty())
                    for (const std::string& n : *names_) matches_.push_back(FilenameMatch{n, false, {}});
                std::set<std::string> times, tiles;
                for (const FilenameMatch& m : matches_) {
                    if (!m.matched) continue;
                    ++matchedCount_;
                    times.insert(group(m, {"t", "time"}));
                    tiles.insert(tileKey(m));
                }
                refreshChannels();
                refreshTileMap();

                if (!r.error.empty()) {
                    status_ = "Pattern error: " + r.error;
                } else if (names_->empty()) {
                    status_ = "The folder holds no TIFF files.";
                } else if (pat.empty()) {
                    status_ = "Enter a pattern or pick a preset.";
                } else {
                    status_ = format("%d of %d file(s) match", matchedCount_, static_cast<int>(names_->size()));
                    if (matchedCount_ > 0)
                        status_ += format(" · %d channel(s) · %d time point(s) · %d tile(s)",
                                          static_cast<int>(std::max<std::size_t>(tokens_.size(), 1)), static_cast<int>(times.size()),
                                          static_cast<int>(tiles.size()));
                }
                if (cluster_ && cluster_->truncated)
                    status_ += format(" · only the first %d TIFF names were listed", static_cast<int>(cluster_->tiffs.size()));
                statusBad_ = !r.error.empty() || (patternOk_ && matchedCount_ == 0);
                canOpen_ = patternOk_ && matchedCount_ > 0;
                if (cluster_ && cluster_->store && names_->empty()) {
                    // a zarr / N5 store names its own axes: it opens as it is
                    status_ = "A zarr / N5 store: it opens as it is, without a pattern.";
                    statusBad_ = false;
                    canOpen_ = true;
                }
            }

            // One row per channel token the pattern found, seeded from the
            // existing manifest (by label, short name or wavelength) or from the
            // token itself.
            void refreshChannels() {
                std::vector<std::string> found;
                std::set<std::string> seen;
                for (const FilenameMatch& m : matches_) {
                    if (!m.matched) continue;
                    const std::string& t = group(m, {"channel", "c"});
                    if (t.empty()) continue;
                    if (seen.insert(t).second) found.push_back(t);
                }
                std::sort(found.begin(), found.end(), tokenLess);
                if (found == tokens_) return;
                tokens_ = found;
                for (std::size_t i = 0; i < tokens_.size(); ++i) {
                    const std::string& token = tokens_[i];
                    if (chans_.find(token) != chans_.end()) continue;
                    Chan ch;
                    ch.info.label = token;
                    double nm = 0.0;
                    if (toNumber(token, nm) && nm > 100.0) ch.info.wavelengthNm = nm;
                    bool seeded = false;
                    if (existing_) {
                        const std::vector<ChannelInfo>& ex = existing_->channels;
                        for (const ChannelInfo& c : ex) {
                            if (c.label == token || c.shortName() == token ||
                                (ch.info.wavelengthNm > 0.0 && std::abs(c.wavelengthNm - ch.info.wavelengthNm) < 0.5)) {
                                ch.info = c;
                                seeded = true;
                                break;
                            }
                        }
                        if (!seeded && ex.size() == tokens_.size()) {
                            ch.info = ex[i];
                            seeded = true;
                        }
                    }
                    if (seeded) ch.customColor = true;
                    else ch.info.color = colorForWavelength(ch.info.wavelengthNm);
                    ch.labelText = ch.info.label;
                    ch.nmText = ch.info.wavelengthNm > 0.0 ? std::to_string(static_cast<int>(std::lround(ch.info.wavelengthNm)))
                                                           : std::string();
                    chans_.emplace(token, std::move(ch));
                }
            }

            void refreshTileMap() {
                const FilenameRule::Positions mode = positionsMode();
                tilePoints_.clear();
                std::set<std::string> tiles;
                bool anyXY = false;
                for (const FilenameMatch& m : matches_) {
                    if (!m.matched) continue;
                    const std::string& x = group(m, {"x", "col"});
                    const std::string& y = group(m, {"y", "row"});
                    if (!tiles.insert(tileKey(m)).second) continue;
                    if (x.empty() && y.empty()) continue;
                    anyXY = true;
                    double vx = 0.0, vy = 0.0;
                    if (!toNumber(x, vx)) vx = 0.0;
                    if (!toNumber(y, vy)) vy = 0.0;
                    tilePoints_.emplace_back(vx, vy);
                }
                const int count = static_cast<int>(tiles.size());
                if (matchedCount_ == 0) {
                    tilePoints_.clear();
                    tileNote_ = "—";
                } else if (mode == FilenameRule::Positions::None) {
                    tilePoints_.clear();
                    tileNote_ = count > 1 ? format("%d tiles, no positions", count) : std::string("single tile");
                } else if (!anyXY) {
                    tilePoints_.clear();
                    tileNote_ = count > 1 ? format("%d tiles · no x / y groups", count) : std::string("single tile");
                } else if (mode == FilenameRule::Positions::GridIndex) {
                    double minx = tilePoints_[0].first, maxx = minx, miny = tilePoints_[0].second, maxy = miny;
                    for (const auto& p : tilePoints_) {
                        minx = std::min(minx, p.first);
                        maxx = std::max(maxx, p.first);
                        miny = std::min(miny, p.second);
                        maxy = std::max(maxy, p.second);
                    }
                    // Checked as doubles: an x group that caught a time stamp
                    // spans more cells than an int holds.
                    const double spanX = maxx - minx, spanY = maxy - miny;
                    constexpr double kMaxCells = 1.0e6;
                    if (!std::isfinite(spanX) || !std::isfinite(spanY) || spanX > kMaxCells || spanY > kMaxCells) {
                        gridNx_ = gridNy_ = 0;
                        tileNote_ = format("%d tiles · grid indices out of range", static_cast<int>(tilePoints_.size()));
                    } else {
                        gridNx_ = static_cast<int>(spanX) + 1;
                        gridNy_ = static_cast<int>(spanY) + 1;
                        tileNote_ = format("%d tiles · %d × %d grid", static_cast<int>(tilePoints_.size()), gridNy_, gridNx_);
                    }
                } else {
                    tileNote_ = format("%d positions (µm)", static_cast<int>(tilePoints_.size()));
                }
                tileGrid_ = mode == FilenameRule::Positions::GridIndex;
            }

            // --- opening -------------------------------------------------------------------

            // Build the mapping from the rule and open it. The sidecar goes
            // where the Manifest field says; beside the TIFFs only once the user
            // has agreed to that, else (or when that folder is not writable)
            // into a local cache with files_folder. An already-valid loaded
            // manifest is opened as-is.
            void saveAndOpen() {
                if (building_ || !canOpen_) return;
                if (remote_) {
                    saveAndOpenOnCluster();
                    return;
                }
                BuildRequest q;
                q.folder = folder_;
                q.destText = trimmed(manifestPath_);
                if (q.destText.empty()) q.destText = folder_ + "/" + DatasetManifest::kFileName;
                q.rule = rule();
                q.fields = {pattern_, positions_, overlap_};
                q.loadedManifestPath = loadedManifestPath_;
                q.existing = existing_;
                building_ = true;
                worker_.run([this, alive = alive_, q = std::move(q)](const Worker::Post& post) {
                    if (!alive->load()) return;
                    BuildResult r = build(q);
                    // not after Cancel, which may have come in the frame this arrived in
                    post([this, alive, r = std::move(r)] {
                        if (alive->load() && isOpen()) built(r);
                    });
                });
            }

            // The same manifest, built from the cluster's listing with the
            // stacks' shapes from the engine on the node, kept in
            // ~/.sirius/manifests there (never in the data folder), and
            // opened by its cluster:// path: the engine reads it as any manifest.
            void saveAndOpenOnCluster() {
                if (!cluster_) return;
                if (cluster_->store && cluster_->tiffs.empty()) {
                    open(cluster_->clusterPath(), {pattern_, positions_, overlap_}, "Another task is still running: cancel it or wait, then open the dataset.");
                    return;
                }
                if (!app_.cluster().connected()) {
                    app_.message(kTitle, "The engine on the node reads the stacks' sizes and opens the dataset: start the worker first "
                                         "(the Cluster button), then Open. The preview needs only the login.");
                    return;
                }
                building_ = true;
                cluster::Session* session = &app_.cluster().session();
                worker_.run([this, alive = alive_, session, folder = cluster_, rule = rule(),
                             fields = RememberedFields{pattern_, positions_, overlap_}](const Worker::Post& post) {
                    if (!alive->load()) return;
                    BuildResult r;
                    r.fields = fields;
                    DatasetManifest manifest;
                    try {
                        manifest = manifestFromClusterFolder(*folder, rule);
                    } catch (const std::exception& e) {
                        r.error = std::string("The files do not form a dataset:\n") + e.what();
                    }
                    if (r.error.empty()) {
                        try {
                            r.openPath = writeClusterManifest(*session, *folder, std::move(manifest));
                        } catch (const ssh::SshError& e) {
                            r.error = std::string("Could not keep the manifest in ~/.sirius/manifests on the cluster:\n") + e.what() +
                                      (e.detail.empty() ? std::string() : "\n" + e.detail);
                        } catch (const std::exception& e) {
                            r.error = std::string("Could not keep the manifest on the cluster:\n") + e.what();
                        }
                    }
                    post([this, alive, r = std::move(r)] {
                        if (alive->load() && isOpen()) built(r);
                    });
                });
            }

            void built(const BuildResult& r) {
                building_ = false;
                if (!r.error.empty()) {
                    app_.message(kTitle, r.error);
                    return;
                }
                if (!r.openPath.empty()) {
                    app_.wb().logLine("Open folder: " + makeClusterPath(host_, folder_) + " -> " + r.openPath +
                                      " (the manifest is kept on the cluster; the data folder is not written)");
                    open(r.openPath, r.fields, "The mapping is ready but another task is still running: cancel it or wait, then open the dataset.");
                    return;
                }
                if (r.openAsIs) {
                    open(fromPath(r.dest), r.fields, "Another task is still running: cancel it or wait, then open the dataset.");
                    return;
                }
                // Nothing is written into a folder, or over a file, the user has
                // not agreed to. The sidecar used to go beside the TIFFs whenever
                // that folder was writable, which on a cluster is source data that
                // merely has group write permission, and over an existing
                // manifest -- a hand-edited one included -- without a word.
                if (r.destExists) {
                    app_.ask(kTitle, fromPath(r.dest) + " exists.\n\nReplace it with the mapping built from this pattern?",
                             {"Yes", "Cancel"}, [this, alive = alive_, r](int answer) {
                                 if (answer == 0 && alive->load()) writeAndOpen(r.manifest, r.dest, r.fields);
                             });
                    return;
                }
                if (r.destInFolder) {
                    const std::string where = settings().getString(kSidecarPlaceKey);   // "data" | "cache" | ""
                    if (where != "data" && where != "cache") {
                        app_.wb().logLine("Open folder: " + folder_ + " has no " + DatasetManifest::kFileName +
                                          " yet; the application asks where to keep it.");
                        auto answered = [this, alive = alive_, r](SidecarQuestion::Answer answer, bool remember) {
                            if (answer == SidecarQuestion::Cancelled || !alive->load()) return;
                            const bool data = answer == SidecarQuestion::Data;
                            if (remember) settings().set(kSidecarPlaceKey, data ? "data" : "cache");
                            placeAndOpen(r, data);
                        };
                        // nobody to ask: the choice that adds nothing to the data folder
                        if (app_.unattended()) answered(SidecarQuestion::Cache, false);
                        else app_.showDialog(std::make_shared<SidecarQuestion>(folder_, answered));
                        return;
                    }
                    placeAndOpen(r, where == "data");
                    return;
                }
                writeAndOpen(r.manifest, r.dest, r.fields);
            }

            void placeAndOpen(const BuildResult& r, bool besideTheFiles) {
                fs::path dest = r.dest;
                if (!besideTheFiles) {
                    dest = cachedManifestPath(canonicalFolder_);
                    manifestPath_ = fromPath(dest);
                }
                writeAndOpen(r.manifest, dest, r.fields);
            }

            void writeAndOpen(DatasetManifest manifest, fs::path dest, const RememberedFields& fields) {
                const fs::path folderPath = toPath(folder_);
                auto writeManifest = [&](const fs::path& path) {
                    if (!samePath(path.parent_path(), folderPath)) manifest.filesFolder = fromPath(canonicalPath(folderPath));
                    else manifest.filesFolder.clear();
                    std::error_code ec;
                    fs::create_directories(path.parent_path(), ec);
                    manifest.save(path);
                };
                try {
                    writeManifest(dest);
                } catch (const std::exception& first) {
                    const fs::path cache = cachedManifestPath(canonicalFolder_);
                    if (samePath(cache, dest)) {
                        app_.message(kTitle, "Could not write " + fromPath(dest) + ":\n" + first.what());
                        return;
                    }
                    try {
                        writeManifest(cache);
                        dest = cache;
                        manifestPath_ = fromPath(dest);
                    } catch (const std::exception& e) {
                        app_.message(kTitle,
                                     std::string("Could not write a manifest (the TIFF folder may not be writable):\n") + e.what());
                        return;
                    }
                }
                open(fromPath(dest), fields, "The mapping is ready but another task is still running: cancel it or wait, then open the dataset.");
            }

            void open(const std::string& path, const RememberedFields& fields, const char* busyText) {
                OpenOptions options;
                options.tile = 0;
                // on the cluster it stays there: the viewer gets what it draws
                options.readAll = remote_ ? false : readAll_;
                // refused while a run or another task is active: the dialog stays
                if (!app_.wb().canEdit() || !bridge_.openDatasetAsync(path, options)) {
                    app_.message(kTitle, busyText);
                    return;
                }
                if (remote_) App::addRecentFile(path);
                // only a pattern that opened something is worth offering again,
                // and it is the one the manifest was built with, not what the
                // fields were edited to while it was being built
                rememberPatternFor(canonicalFolder_, fields.pattern, fields.positions, fields.overlap);
                if (opened_) opened_();
                close();
            }

            // --- drawing ---------------------------------------------------------------------

            void drawFolderRows(App& app) {
                const float scale = std::max(theme::scale(), 0.01f);
                const float spacing = px(8);
                ClusterLink& link = app.cluster();
                if (link.sshUp() || remote_) {
                    // where the folder is: this computer, or the cluster (read there by the engine)
                    int where = remote_ ? 1 : 0;
                    widgets::SegmentedOpts so;
                    so.enabled = !building_;
                    so.tooltips = {"A folder on this computer",
                                   "A folder on the cluster: listed over the SSH session, read on the node by the engine; the manifest "
                                   "is kept in ~/.sirius/manifests there"};
                    if (widgets::segmented("##where", {"This computer", "Cluster"}, &where, so)) {
                        const bool remote = where == 1;
                        app.defer([this, alive = alive_, remote] {
                            if (alive->load() && isOpen()) setWhere(remote);
                        });
                    }
                    if (remote_) {
                        ImGui::SameLine(0.0f, px(10));
                        // centred on the switch's row
                        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + std::floor((theme::snap(px(26)) - theme::textSize("Ag", 11).y) * 0.5f));
                        if (!link.sshUp()) widgets::text("Not logged in to the cluster", 11, theme::kAccentText);
                        else if (!link.connected())
                            widgets::text(host_ + " · the preview works now; Open waits for the worker", 11, theme::kNeutral600);
                        else widgets::text(host_ + " · names listed over SSH, files read on the node", 11, theme::kNeutral600);
                    }
                    gap(6);
                }
                {
                    // folder + Browse + file count
                    std::string count = names_ ? format("%d TIFF file(s)", static_cast<int>(names_->size()))
                                               : (folder_.empty() ? std::string() : std::string("listing…"));
                    if (cluster_ && cluster_->truncated) count += "+";
                    const float countW = count.empty() ? 0.0f : theme::textSize(count, 12).x;
                    const ImVec2 browseSize = buttonSize("Browse…", widgets::ButtonKind::Secondary, true);
                    const Line line;
                    line.at(0.0f, line.height());
                    widgets::FieldOpts f;
                    f.width = std::max(px(80), line.width() - countW - browseSize.x - 2 * spacing) / scale;
                    f.readOnly = true;
                    f.hint = remote_ ? "a folder on the cluster…" : "a folder…";
                    std::string shown = remote_ && !folder_.empty() ? host_ + ":" + folder_ : folder_;
                    widgets::inputText("##folder", &shown, f);
                    widgets::tooltip(remote_ && !folder_.empty() ? makeClusterPath(host_, folder_) : folder_);
                    const float browseX = line.width() - countW - (countW > 0.0f ? spacing : 0.0f) - browseSize.x;
                    line.at(browseX, browseSize.y);
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.enabled = !building_;
                    b.tooltip = remote_ ? "The cluster's folders, through the SSH session" : "Choose the folder of TIFF stacks";
                    if (widgets::button("Browse…##folder", b)) {
                        app.defer([this, alive = alive_, &app] {
                            if (alive->load() && isOpen()) browse(app);
                        });
                    }
                    if (countW > 0.0f) line.text(line.width() - countW, count, 12, theme::kNeutral600);
                    line.end();
                }
                if (remote_) {
                    // nothing is written into the data folder on the cluster: the manifest is kept with the user's own files there
                    const std::string label = "Manifest";
                    const float labelW = theme::textSize(label, 11).x;
                    const ImVec2 load = buttonSize("Load…", widgets::ButtonKind::Secondary, true);
                    const Line line;
                    line.text(0.0f, label, 11, theme::kNeutral700);
                    const std::string where = "kept in ~/.sirius/manifests on " + (host_.empty() ? std::string("the cluster") : host_) +
                                              "; the data folder is not written";
                    line.text(labelW + spacing, widgets::elideText(where, line.width() - labelW - load.x - 3 * spacing, 12), 12,
                              theme::kNeutral600);
                    line.at(line.width() - load.x, load.y);
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.tooltip = "Use the pattern and metadata of a dataset manifest on this computer";
                    if (widgets::button("Load…", b)) {
                        app.defer([this, alive = alive_, &app] {
                            if (alive->load()) loadManifestFile(app);
                        });
                    }
                    line.end();
                    return;
                }
                {
                    const std::string label = "Manifest";
                    const float labelW = theme::textSize(label, 11).x;
                    const ImVec2 load = buttonSize("Load…", widgets::ButtonKind::Secondary, true);
                    const ImVec2 save = buttonSize("Save as…", widgets::ButtonKind::Secondary, true);
                    const Line line;
                    line.text(0.0f, label, 11, theme::kNeutral700);
                    const float fieldW = std::max(px(80), line.width() - labelW - load.x - save.x - 3 * spacing);
                    line.at(labelW + spacing, line.height());
                    widgets::FieldOpts f;
                    f.width = fieldW / scale;
                    widgets::inputText("##manifest", &manifestPath_, f);
                    widgets::tooltip("Optional. Load an existing .toml, or choose where to write one. Leave the default: "
                                     "Open writes it next to the TIFFs when that folder is writable, otherwise a local cache.");
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.tooltip = "Open an existing dataset manifest and use its filename pattern";
                    float x = labelW + spacing + fieldW + spacing;
                    line.at(x, load.y);
                    if (widgets::button("Load…", b)) {
                        app.defer([this, alive = alive_, &app] {
                            if (alive->load()) loadManifestFile(app);
                        });
                    }
                    x += load.x + spacing;
                    b.tooltip = "Choose where to write a new manifest. Opens outside the TIFF folder so the dialog does not list "
                                "hundreds of stacks";
                    line.at(x, save.y);
                    if (widgets::button("Save as…", b)) {
                        app.defer([this, alive = alive_] {
                            if (alive->load()) saveManifestAs();
                        });
                    }
                    line.end();
                }
            }

            void drawPattern() {
                const float scale = std::max(theme::scale(), 0.01f);
                gap(12);
                widgets::rule(2);
                gap(10);
                widgets::caption("Filename pattern");
                gap(8);
                {
                    const ImVec2 presets = buttonSize(presetLabel_, widgets::ButtonKind::Secondary, true);
                    const float spacing = px(8);
                    const Line line;
                    line.at(0.0f, line.height());
                    widgets::FieldOpts f;
                    f.width = std::max(px(80), line.width() - presets.x - spacing) / scale;
                    f.monospace = true;
                    f.hint = kPresets[0].pattern;
                    if (widgets::inputText("##pattern", &pattern_, f)) patternEdited();
                    widgets::tooltip("Regular expression matched against each file name; named groups pick out the channel, time "
                                     "point and tile");
                    line.at(line.width() - presets.x, presets.y);
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.tooltip = "Common layouts; pick one and adjust";
                    if (widgets::button((presetLabel_ + "##presets").c_str(), b)) ImGui::OpenPopup("##presetMenu");
                    const ImVec2 below(ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y);
                    drawPresetMenu(below);
                    line.end();
                }
                gap(8);
                widgets::textWrapped("Named groups, written (?P<name>…): channel · t · tile · x · y · z. x / y / z are grid indices or "
                                     "stage coordinates (see Positions); files the pattern does not match are left out.",
                                     11, theme::kNeutral600);
                gap(8);
                widgets::textWrapped(status_, 12, statusBad_ ? theme::kAccentText : theme::kNeutral600);
            }

            void drawPresetMenu(ImVec2 below) {
                ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
                ImGui::PushStyleColor(ImGuiCol_Header, theme::kSurface);
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
                ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 2));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
                float width = 0.0f;
                for (const Preset& p : kPresets) width = std::max(width, theme::textSize(p.label, 12).x);
                width += px(24);
                ImGui::SetNextWindowPos(ImVec2(below.x, below.y + px(2)), ImGuiCond_Appearing);
                int picked = -1;
                if (ImGui::BeginPopup("##presetMenu")) {
                    const float h = theme::snap(px(26));
                    for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i) {
                        const Preset& p = kPresets[i];
                        ImGui::PushID(i);
                        const ImVec2 at = ImGui::GetCursorScreenPos();
                        const bool current = pattern_ == p.pattern;
                        if (ImGui::Selectable("##preset", current, ImGuiSelectableFlags_None, ImVec2(width, h))) picked = i;
                        widgets::tooltip(p.example);
                        widgets::drawTextIn(ImGui::GetWindowDrawList(), ImVec2(at.x + px(10), at.y), ImVec2(at.x + width, at.y + h), p.label,
                                            12, theme::kText, current ? theme::Weight::SemiBold : theme::Weight::Regular, 0.0f, 0.5f);
                        ImGui::PopID();
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar(3);
                ImGui::PopStyleColor(5);
                if (picked >= 0) applyPreset(picked);
            }

            // Text in a table cell, centred on the row and, when asked, in the column.
            static void cellText(const std::string& s, float height, float fontPx, ImU32 color, bool centred, bool isCaption = false) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float w = ImGui::GetContentRegionAvail().x;
                ImGui::Dummy(ImVec2(0.0f, height));
                if (s.empty()) return;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (isCaption) {
                    const std::string& t = s;   // written as the header says it, "WAVELENGTH (nm)"
                    const theme::FontScope f(fontPx, theme::captionFont());
                    const ImVec2 ts = ImGui::CalcTextSize(t.c_str());
                    const float x = centred ? p.x + std::max(0.0f, (w - ts.x) * 0.5f) : p.x;
                    dl->AddText(ImVec2(theme::snap(x), theme::snap(p.y + (height - ts.y) * 0.5f)), color, t.c_str());
                    return;
                }
                const ImVec2 ts = theme::textSize(s, fontPx);
                const float x = centred ? p.x + std::max(0.0f, (w - ts.x) * 0.5f) : p.x;
                widgets::drawText(dl, ImVec2(x, p.y + (height - ts.y) * 0.5f), s, fontPx, color);
            }

            void drawPreview() {
                gap(8);
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                const float height = std::max(px(140), ImGui::GetContentRegionAvail().y - lowerHeight_ - spacing);
                const float rowH = theme::snap(px(24));
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, px(8, 0));
                ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, theme::kDivider);
                const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_NoSavedSettings |
                                              ImGuiTableFlags_SizingFixedFit;
                if (ImGui::BeginTable("##preview", 7, flags, ImVec2(0.0f, height))) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("FILE", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("CHANNEL", ImGuiTableColumnFlags_WidthFixed, px(76));
                    ImGui::TableSetupColumn("T", ImGuiTableColumnFlags_WidthFixed, px(44));
                    ImGui::TableSetupColumn("TILE", ImGuiTableColumnFlags_WidthFixed, px(52));
                    ImGui::TableSetupColumn("X", ImGuiTableColumnFlags_WidthFixed, px(52));
                    ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthFixed, px(52));
                    ImGui::TableSetupColumn("Z", ImGuiTableColumnFlags_WidthFixed, px(52));
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                    const char* heads[7] = {"FILE", "CHANNEL", "T", "TILE", "X", "Y", "Z"};
                    for (int c = 0; c < 7; ++c) {
                        ImGui::TableSetColumnIndex(c);
                        cellText(heads[c], rowH, theme::kCaptionPx, theme::kNeutral600, c > 0, true);
                    }
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, theme::kSurface);

                    ImGuiListClipper clipper;
                    clipper.Begin(static_cast<int>(matches_.size()), rowH);
                    while (clipper.Step()) {
                        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                            const FilenameMatch& m = matches_[static_cast<std::size_t>(r)];
                            static const std::string noMatch = "no match";
                            const std::string* cells[7] = {&m.file,
                                                           m.matched ? &group(m, {"channel", "c"}) : &noMatch,
                                                           &group(m, {"t", "time"}),
                                                           &group(m, {"tile"}),
                                                           &group(m, {"x", "col"}),
                                                           &group(m, {"y", "row"}),
                                                           &group(m, {"z"})};
                            const ImU32 color = m.matched ? theme::kText : theme::kNeutral500;
                            ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                            for (int c = 0; c < 7; ++c) {
                                ImGui::TableSetColumnIndex(c);
                                if (c == 0) {
                                    const float room = ImGui::GetContentRegionAvail().x;
                                    const std::string shown = widgets::elideText(m.file, room, 12);
                                    const ImVec2 at = ImGui::GetCursorScreenPos();
                                    cellText(shown, rowH, 12, color, false);
                                    if (shown != m.file && ImGui::IsMouseHoveringRect(at, ImVec2(at.x + room, at.y + rowH))) {
                                        ImGui::SetCursorScreenPos(at);
                                        ImGui::Dummy(ImVec2(room, rowH));
                                        widgets::tooltip(m.file);
                                    }
                                } else {
                                    cellText(widgets::elideText(*cells[c], ImGui::GetContentRegionAvail().x, 12), rowH, 12, color, true);
                                }
                            }
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
            }

            void drawPositions() {
                gap(12);
                widgets::rule(2);
                gap(10);
                const Line line;
                const std::string caption = captionCase("Positions");
                float x = 0.0f;
                {
                    const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
                    const ImVec2 ts = ImGui::CalcTextSize(caption.c_str());
                    line.at(x, ts.y);
                    x += ts.x + px(12);
                }
                widgets::caption("Positions");
                const std::vector<std::string> options{"None", "Grid indices", "Micrometres"};
                widgets::SegmentedOpts s;
                s.tooltips = {"One tile, or tiles without known positions", "x / y / z are column, row and layer indices of a grid",
                              "x / y / z are stage coordinates in micrometres"};
                line.at(x, theme::snap(px(26)));
                if (widgets::segmented("##positions", options, &positions_, s)) {
                    touched_ = true;
                    refreshTileMap();
                }
                x += widgets::segmentedWidth(options) + px(12);
                line.text(x, "Overlap", 11, theme::kNeutral700);
                x += theme::textSize("Overlap", 11).x + px(12);
                line.at(x, line.height());
                if (unitField("##overlap", &overlap_, 0.0, 90.0, 1, "%", px(96), positions_ == 1)) touched_ = true;
                tooltipEvenDisabled("Grid indices: neighbouring tiles overlap by this fraction of their size");
                line.end();
            }

            void drawMetadata() {
                const float scale = std::max(theme::scale(), 0.01f);
                const float spacing = px(10);
                gap(10);
                widgets::caption("Metadata");
                gap(6);
                const float column = px(112);
                Columns cols;
                const float width = cols.width();
                voxelFields(cols, &voxel_[0], &voxel_[1], &voxel_[2], column, spacing);

                cols.labelled(3 * (column + spacing), "Frame interval");
                unitField("##interval", &interval_, 0.0, 1.0e6, 3, "s", column);
                if (interval_ == 0.0 && !ImGui::IsItemActive()) {
                    // the spin box's special value: 0 reads "unknown" until it is edited
                    const float border = theme::crispPen(theme::kBorder);
                    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
                    const ImVec2 a(min.x + border, min.y + border), b(max.x - border, max.y - border);
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(a, b, theme::kBg);
                    widgets::drawTextIn(dl, ImVec2(min.x + px(8), a.y), b, "unknown", 13, theme::kText, theme::Weight::Regular, 0.0f, 0.5f);
                }
                widgets::tooltip("Time between consecutive time points; 0 = unknown");
                cols.track();

                cols.labelled(4 * (column + spacing), "Acquisition");
                widgets::FieldOpts f;
                f.width = std::max(px(80), width - 4 * (column + spacing)) / scale;
                f.hint = "widefield, 3D-SIM, confocal…";
                widgets::inputText("##acquisition", &acquisition_, f);
                cols.track();
                cols.end();

                gap(8);
                simRow(sim_);
            }

            void drawChannelsAndTiles() {
                gap(12);
                widgets::rule(2);
                gap(10);
                const float mapW = theme::snap(px(188)), mapH = theme::snap(px(132));
                const float spacing = px(16);
                const float left = std::max(px(120), ImGui::GetContentRegionAvail().x - mapW - spacing);

                Columns cols;
                cols.at(0.0f);
                ImGui::BeginGroup();
                widgets::caption("Channels");
                gap(6);
                if (tokens_.empty()) {
                    widgets::textWrapped("No channel group in the pattern: the files hold one channel.", 11, theme::kNeutral600,
                                         theme::Weight::Regular, left);
                } else {
                    drawChannelTable(left);
                }
                ImGui::EndGroup();
                cols.track();

                cols.at(left + spacing);
                ImGui::BeginGroup();
                widgets::caption("Tiles");
                gap(6);
                drawTileMap(ImVec2(mapW, mapH));
                ImGui::EndGroup();
                cols.track();
                cols.end();
            }

            void drawChannelTable(float width) {
                const float rowH = theme::snap(px(26));
                const float height = std::min(px(132), rowH * static_cast<float>(tokens_.size() + 1) + px(4));
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, px(8, 0));
                ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, theme::kDivider);
                const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_NoSavedSettings |
                                              ImGuiTableFlags_SizingFixedFit;
                if (ImGui::BeginTable("##channels", 4, flags, ImVec2(width, height))) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("TOKEN", ImGuiTableColumnFlags_WidthFixed, px(70));
                    ImGui::TableSetupColumn("LABEL", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("WAVELENGTH (nm)", ImGuiTableColumnFlags_WidthFixed, px(116));
                    ImGui::TableSetupColumn("COLOUR", ImGuiTableColumnFlags_WidthFixed, px(60));
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                    const char* heads[4] = {"TOKEN", "LABEL", "WAVELENGTH (nm)", "COLOUR"};
                    for (int c = 0; c < 4; ++c) {
                        ImGui::TableSetColumnIndex(c);
                        cellText(heads[c], rowH, theme::kCaptionPx, theme::kNeutral600, c >= 2, true);
                    }
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, theme::kSurface);
                    for (std::size_t i = 0; i < tokens_.size(); ++i) {
                        const auto it = chans_.find(tokens_[i]);
                        if (it == chans_.end()) continue;
                        Chan& ch = it->second;
                        ImGui::PushID(static_cast<int>(i));
                        ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                        ImGui::TableSetColumnIndex(0);
                        cellText(widgets::elideText(tokens_[i], ImGui::GetContentRegionAvail().x, 13), rowH, 13, theme::kNeutral600, false);
                        ImGui::TableSetColumnIndex(1);
                        if (cellEdit("##label", &ch.labelText, rowH, cellEditing_)) ch.info.label = trimmed(ch.labelText);
                        ImGui::TableSetColumnIndex(2);
                        if (cellEdit("##nm", &ch.nmText, rowH, cellEditing_)) {
                            double nm = 0.0;
                            ch.info.wavelengthNm = toNumber(ch.nmText, nm) && nm > 0.0 ? nm : 0.0;
                            if (!ch.customColor) ch.info.color = colorForWavelength(ch.info.wavelengthNm);
                        }
                        ImGui::TableSetColumnIndex(3);
                        drawColourCell(tokens_[i], ch, rowH);
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
            }

            // A text field that looks like the cell it is in until it has the
            // keyboard. `editing` is set while it has it, and in the frame Enter
            // commits it: that Enter is the cell's, not the Open button's.
            static bool cellEdit(const char* id, std::string* value, float rowH, bool& editing) {
                const theme::FontScope font(13);
                const float fh = ImGui::GetFontSize();
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, std::max(0.0f, std::floor((rowH - fh) * 0.5f))));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, theme::kTransparent);
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, theme::kNeutral200);
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, theme::kBg);
                ImGui::SetNextItemWidth(-FLT_MIN);
                const bool changed = ImGui::InputText(id, value);
                if (ImGui::IsItemActive() || ImGui::IsItemDeactivated()) editing = true;
                if (ImGui::IsItemActive()) {
                    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(min.x, max.y - theme::crispPen(2)), max, theme::kAccent);
                }
                ImGui::PopStyleColor(3);
                ImGui::PopStyleVar(2);
                widgets::tooltip("Click a label or wavelength to edit; click the colour to change it");
                return changed;
            }

            void drawColourCell(const std::string& token, Chan& ch, float rowH) {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float w = std::max(1.0f, ImGui::GetContentRegionAvail().x);
                if (ImGui::InvisibleButton("##colour", ImVec2(w, rowH))) ImGui::OpenPopup("##picker");
                const bool hovered = ImGui::IsItemHovered();
                if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                widgets::tooltip("Display colour");
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (hovered) dl->AddRectFilled(at, ImVec2(at.x + w, at.y + rowH), theme::kNeutral200);
                const float chip = theme::snap(px(14));
                const ImVec2 a(theme::snap(at.x + (w - chip) * 0.5f), theme::snap(at.y + (rowH - chip) * 0.5f));
                dl->AddRectFilled(a, ImVec2(a.x + chip, a.y + chip), theme::fromFloat(ch.info.color));

                ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
                ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(12, 10));
                if (ImGui::BeginPopup("##picker")) {
                    widgets::text("Colour for channel " + token, 12, theme::kText, theme::Weight::SemiBold);
                    {   // the font is popped inside the popup it was pushed in
                        const theme::FontScope font(12);
                        ImGui::SetNextItemWidth(px(220));
                        if (ImGui::ColorPicker3("##rgb", ch.info.color.data(),
                                                ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview |
                                                    ImGuiColorEditFlags_DisplayHex | ImGuiColorEditFlags_NoLabel))
                            ch.customColor = true;
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }

            // The tiles as the pattern places them: grid cells or scaled stage
            // positions, with a one-line summary underneath.
            void drawTileMap(ImVec2 size) const {
                const float s = theme::scale();
                const ImVec2 o = ImGui::GetCursorScreenPos();
                ImGui::Dummy(size);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const ImVec2 end(o.x + size.x, o.y + size.y);
                dl->AddRectFilled(o, end, theme::kSurface);
                widgets::crispRect(dl, o, end, theme::kDivider, 1.0f);
                widgets::drawTextIn(dl, ImVec2(o.x, end.y - 20.0f * s), ImVec2(end.x, end.y - 2.0f * s), tileNote_, 11, theme::kNeutral600);
                if (tilePoints_.empty()) return;
                double minx = tilePoints_[0].first, maxx = minx, miny = tilePoints_[0].second, maxy = miny;
                for (const auto& q : tilePoints_) {
                    minx = std::min(minx, q.first);
                    maxx = std::max(maxx, q.first);
                    miny = std::min(miny, q.second);
                    maxy = std::max(maxy, q.second);
                }
                // the area the tiles are drawn in, design pixels from the corner
                const double left = 10.0, top = 10.0, width = size.x / s - 20.0, height = size.y / s - 36.0;
                auto point = [&](double x, double y) {
                    return ImVec2(theme::snap(o.x + static_cast<float>(x) * s), theme::snap(o.y + static_cast<float>(y) * s));
                };
                if (tileGrid_) {
                    const int nx = gridNx_, ny = gridNy_;   // 0: out of range, which the note says
                    if (nx <= 0 || ny <= 0) return;
                    const double cell = std::clamp(std::min(width / nx, height / ny), 4.0, 26.0);
                    const double ox = left + (width - cell * nx) / 2.0;
                    const double oy = top + (height - cell * ny) / 2.0;
                    dl->PushClipRect(o, end, true);
                    for (const auto& q : tilePoints_) {
                        const double x = ox + (q.first - minx) * cell, y = oy + (q.second - miny) * cell;
                        const ImVec2 a = point(x + 1.0, y + 1.0), b = point(x + cell - 1.0, y + cell - 1.0);
                        dl->AddRectFilled(a, b, theme::withAlpha(theme::kAccent, 40.0f / 255.0f));
                        widgets::crispRect(dl, a, b, theme::kAccent, 1.0f);
                    }
                    dl->PopClipRect();
                } else {
                    const double sx = maxx > minx ? (width - 8.0) / (maxx - minx) : 0.0;
                    const double sy = maxy > miny ? (height - 8.0) / (maxy - miny) : 0.0;
                    double k = 0.0;
                    if (sx > 0.0 && sy > 0.0) k = std::min(sx, sy);
                    else k = std::max(sx, sy);
                    const double w = k * (maxx - minx), h = k * (maxy - miny);
                    const double ox = left + (width - w) / 2.0, oy = top + (height - h) / 2.0;
                    for (const auto& q : tilePoints_) {
                        const double x = ox + (q.first - minx) * k, y = oy + (q.second - miny) * k;
                        dl->AddRectFilled(point(x - 3.0, y - 3.0), point(x + 3.0, y + 3.0), theme::kAccent);
                    }
                }
            }

            void drawButtons() {
                gap(14);
                const bool enabled = canOpen_ && !building_ && !matching_ && !previewAt_;
                const int pressed = footer({{"Cancel", widgets::ButtonKind::Ghost, true, {}},
                                            {building_ ? "Opening…" : "Open", widgets::ButtonKind::Primary, enabled,
                                             "Open the folder as a dataset. Remembers this filename pattern for next time."}});
                if (pressed == 0) close();
                else if (pressed == 1 || (enabled && enterPressed(popupAtStart_, cellEditing_))) saveAndOpen();
            }

            App& app_;
            Bridge& bridge_;
            std::function<void()> opened_;                // the Open dialog under this one, when it raised it
            bool readAll_ = true;                         // full load, or lazy: that dialog's "Read as"
            bool remote_ = false;                         // folder_ is on the cluster (host_)
            std::string host_;
            std::string folder_;                          // "" until one is chosen
            std::string canonicalFolder_;                 // on the cluster: its cluster:// path
            std::shared_ptr<const ClusterFolder> cluster_;   // the cluster folder as listed
            std::uint64_t listGeneration_ = 0;
            Names names_;                                 // null until the folder is listed
            std::optional<DatasetManifest> existing_;     // a manifest already in the folder or loaded
            std::string loadedManifestPath_;              // toml we loaded; empty if none

            std::string pattern_;
            std::string presetLabel_ = "Presets";
            std::string manifestPath_;
            std::string status_;
            bool statusBad_ = false;
            int positions_ = 1;
            double overlap_ = 10.0;
            std::array<double, 3> voxel_{0.1, 0.1, 0.2};
            double interval_ = 0.0;
            std::string acquisition_;
            SimFields sim_;
            bool touched_ = false;                        // edited before the folder was listed
            bool popupAtStart_ = false;                   // this frame's Enter went to a popup
            bool cellEditing_ = false;                    // ... or to a cell of the channel table

            std::optional<Clock::time_point> previewAt_;
            std::uint64_t generation_ = 0;
            bool matching_ = false;
            bool building_ = false;
            std::vector<FilenameMatch> matches_;
            bool patternOk_ = false;
            int matchedCount_ = 0;
            bool canOpen_ = false;

            std::map<std::string, Chan> chans_;           // by channel token
            std::vector<std::string> tokens_;             // channel table rows, in order
            std::vector<std::pair<double, double>> tilePoints_;
            bool tileGrid_ = true;
            int gridNx_ = 0, gridNy_ = 0;                 // grid indices: the map's columns and rows, 0 = out of range
            std::string tileNote_ = "—";
            float lowerHeight_ = 430.0f;                  // of what lies below the preview, as last laid out

            Alive alive_ = makeAlive();
            Worker worker_;
        };

    } // namespace

    std::shared_ptr<Dialog> makeFolderDatasetDialog(App& app, const std::string& folder) {
        return std::make_shared<FolderDatasetDialog>(app, folder, nullptr, true);
    }

    std::shared_ptr<Dialog> dataset_dialogs::makeFolderDatasetDialog(App& app, const std::string& folder, std::function<void()> opened,
                                                                     bool readAll) {
        return std::make_shared<FolderDatasetDialog>(app, folder, std::move(opened), readAll);
    }

} // namespace sirius::app::gui
