#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <functional>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_stdlib.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "core/ops/plugin.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/code_editor.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

namespace sirius::app::gui {

    namespace fs = std::filesystem;
    using theme::px;
    using theme::Weight;

    namespace {

        enum Status { kDirectory = 0,
                      kMissingDirectory,
                      kLoaded,
                      kFailed,
                      kSkipped };

        constexpr float kDirRowHeight = 26;
        constexpr float kFileRowHeight = 28;
        constexpr float kErrorRowHeight = 44;
        constexpr float kLeftWidth = 300;

        // One row of the list: a folder caption, or a file with its status.
        struct Row {
            std::string path;       // file path (file rows) or directory path (directory rows)
            std::string text;       // shown name
            std::string kind;       // operation kind, "helper", "not loaded"
            std::string error;      // first line of the load error
            std::string tip;
            Status status = kDirectory;
            bool isDir() const { return status <= kMissingDirectory; }
        };

        std::string clean(const std::string& p) {
            std::error_code ec;
            fs::path path = fs::absolute(fs::u8path(p), ec);
            if (ec) path = fs::u8path(p);
            std::string s = path.lexically_normal().generic_u8string();
            while (s.size() > 1 && s.back() == '/' && !(s.size() == 3 && s[1] == ':')) s.pop_back();
            return s;
        }
        std::string cleanDir(const std::string& dir) { return clean(dir); }
        std::string cleanFile(const std::string& file) { return clean(file); }
        std::string dirOf(const std::string& file) { return clean(fs::u8path(file).parent_path().u8string()); }

        bool lessNoCase(const std::string& a, const std::string& b) { return toLower(a) < toLower(b); }

        std::string identifier(const std::string& raw) {
            const std::string name = toLower(trimmed(raw));
            std::string out;
            bool gap = false;
            for (char c : name) {
                const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
                if (ok) {
                    if (gap) out += '_';
                    gap = false;
                    out += c;
                } else {
                    gap = true;   // a run of anything else becomes one '_'
                }
            }
            while (!out.empty() && out.front() == '_') out.erase(out.begin());
            while (!out.empty() && out.back() == '_') out.pop_back();
            return out;
        }

        std::string titleCase(const std::string& kind) {
            std::vector<std::string> words = split(kind, '_', true);
            for (std::string& w : words) w[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(w[0])));
            return join(words, " ");
        }

        std::string pluginTemplate(const std::string& kind, const std::string& title) {
            std::string t = R"PY("""%TITLE%: a SIRIUS user operation.

Edit STEP (the name and the parameters the step card shows) and run() (what
it does). Save reloads the plugin; a file that fails to load shows its error
here and in the add menu. plugins/README.md beside the app has the full
contract.
"""

import numpy as np

STEP = {
    "kind": "%KIND%",          # unique id, stored in saved pipelines
    "name": "%TITLE%",         # shown in the add menu and on the step card
    "group": "User",
    "params": [
        {"key": "offset", "label": "Offset", "type": "double", "default": 0.0, "min": -1e6, "max": 1e6,
         "unit": "counts", "help": "Value subtracted from every voxel"},
        {"key": "clip", "label": "Clip negatives", "type": "bool", "default": True,
         "help": "Set values below 0 to 0"},
    ],
    "separable_over_t": True,   # frames are independent: the app may split the run over t
}


def run(data, params, meta, ctx):
    """# %TITLE%

    Markdown shown as the operation's help. `data` is a float32 array of
    shape (c, t, z, y, x); `meta` carries dims and voxel size; return an
    array with the same layout plus a dict with a one-line "summary" and
    optional "facts" for the step card.
    """
    offset = float(params["offset"])
    c, t = data.shape[:2]
    out = np.empty_like(data, dtype=np.float32)
    n = c * t
    k = 0
    for ci in range(c):
        for ti in range(t):
            if ctx.cancelled():
                raise RuntimeError("cancelled")
            vol = data[ci, ti]
            # --- the operation: replace with your own ---
            out[ci, ti] = vol - offset
            k += 1
            ctx.progress(k / n, f"channel {ci} t {ti}")
    if params["clip"]:
        np.maximum(out, 0.0, out=out)
    return out, {"summary": f"%TITLE% offset {offset:g}", "facts": {"Offset": f"{offset:g} counts"}}
)PY";
            t = replaceAll(t, "%KIND%", kind);
            return replaceAll(t, "%TITLE%", title);
        }

        // --- text in a given face ------------------------------------------------

        float fontSize(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return designPx * st.FontScaleMain * st.FontScaleDpi;
        }

        float measure(ImFont* font, float designPx, const std::string& s) {
            if (!font) font = ImGui::GetFont();
            return font->CalcTextSizeA(fontSize(designPx), FLT_MAX, 0.0f, s.c_str(), s.c_str() + s.size()).x;
        }

        // Cut in the middle, so the file name at the end stays readable.
        std::string elideMiddle(const std::string& s, float width, ImFont* font, float designPx) {
            if (measure(font, designPx, s) <= width) return s;
            std::vector<std::size_t> starts;
            for (std::size_t i = 0; i < s.size();) {
                starts.push_back(i);
                nextCodepoint(s, i);
            }
            const std::string dots = "\xE2\x80\xA6";
            std::size_t keep = starts.size();
            while (keep > 1) {
                --keep;
                const std::size_t headN = keep / 2, tailN = keep - headN;
                const std::string head = s.substr(0, starts[headN]);
                const std::string tail = s.substr(starts[starts.size() - tailN]);
                const std::string candidate = head + dots + tail;
                if (measure(font, designPx, candidate) <= width) return candidate;
            }
            return dots;
        }

        std::string elideRight(const std::string& s, float width, ImFont* font, float designPx) {
            if (measure(font, designPx, s) <= width) return s;
            const std::string dots = "\xE2\x80\xA6";
            std::vector<std::size_t> ends;
            for (std::size_t i = 0; i < s.size();) {
                nextCodepoint(s, i);
                ends.push_back(i);
            }
            for (std::size_t n = ends.size(); n-- > 0;) {
                const std::string candidate = s.substr(0, ends[n]) + dots;
                if (measure(font, designPx, candidate) <= width) return candidate;
            }
            return dots;
        }

        void drawIn(ImDrawList* dl, ImFont* font, float designPx, ImVec2 pos, ImU32 color, const std::string& s) {
            if (!font) font = ImGui::GetFont();
            dl->AddText(font, fontSize(designPx), ImVec2(theme::snap(pos.x), theme::snap(pos.y)), color, s.c_str(),
                        s.c_str() + s.size());
        }

        float lineHeight(ImFont* font, float designPx) {
            if (!font) font = ImGui::GetFont();
            return font->CalcTextSizeA(fontSize(designPx), FLT_MAX, 0.0f, "Ag").y;
        }

        bool readWhole(const std::string& path, std::string& out, std::string& error) {
            std::ifstream in(fs::u8path(path), std::ios::binary);
            if (!in) {
                std::error_code ec(errno, std::generic_category());
                error = ec.message();
                return false;
            }
            std::stringstream ss;
            ss << in.rdbuf();
            out = ss.str();
            return true;
        }

    } // namespace

    class PluginManagerDialog final : public PluginManager, public std::enable_shared_from_this<PluginManagerDialog> {
    public:
        explicit PluginManagerDialog(App& app) : app_(app) {
            editor_.setPlaceholder("Select a file on the left, or click New.");
            editor_.setReadOnly(true);
            // What the last load of the plugins said (the worker may not start):
            // the session's log so far, then every new line.
            for (const std::string& line : app.wb().log()) notePluginLog(line);
            logConnection_ = app.bridge().logged.connect([this](const std::string& line) { notePluginLog(line); });
            seenOperations_ = app.bridge().rev().operations;
            refreshTree();
        }

        ~PluginManagerDialog() override { app_.bridge().logged.disconnect(logConnection_); }

        std::string title() const override { return "User operations"; }
        ImVec2 size() const override { return ImVec2(900, 600); }
        bool modal() const override { return false; }
        bool resizable() const override { return true; }

        bool canClose(App&) override {
            if (!modified_ || currentPath_.empty()) return true;
            confirmDiscard([w = weak_from_this()] {
                if (auto self = w.lock()) self->close();
            });
            return false;
        }

        void closed(App&) override { refreshOnShow_ = true; }

        void openFile(const std::string& path) override {
            const std::string c = cleanFile(path);
            int row = rowForPath(c);
            if (row < 0) {
                refreshTree();
                row = rowForPath(c);
            }
            if (row >= 0) {
                selectRow(row);   // loads the file (asks about unsaved edits first)
                scrollTo_ = c;
                return;
            }
            std::error_code ec;
            if (c == currentPath_ || !fs::exists(fs::u8path(c), ec)) return;
            confirmDiscard([w = weak_from_this(), c] {
                if (auto self = w.lock()) {
                    self->current_ = -1;
                    self->loadFile(c);
                }
            });
        }

        void draw(App& app) override {
            if (refreshOnShow_ || seenOperations_ != app.bridge().rev().operations) {
                seenOperations_ = app.bridge().rev().operations;
                refreshOnShow_ = false;
                refreshTree();
            }
            const bool focusedHere = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            if (focusedHere) {
                // The main window's shortcuts (Backspace removes a step,
                // Ctrl+S saves the pipeline) would reach past this dialog,
                // so while it has the keyboard the application is told that
                // text is being typed, which keeps its shortcuts off.
                ImGui::GetCurrentContext()->PlatformImeData.WantTextInput = true;
                if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S)) {
                    if (!currentPath_.empty() && !editor_.readOnly() && modified_) save();
                } else if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && !editor_.focused() && !ImGui::IsAnyItemActive() &&
                           !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
                    if (canClose(app)) close();
                }
            }

            widgets::textWrapped("Python files in these folders become operations in the add menu. Save reloads them.",
                                 theme::kSmallPx, theme::kNeutral600);
            widgets::vspace(8);
            widgets::rule(2);
            widgets::vspace(10);

            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float leftW = std::min(px(kLeftWidth), avail.x * 0.5f);
            const float h = std::max(px(120), avail.y);

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
            ImGui::BeginGroup();
            drawLeft(ImVec2(leftW - px(14), h));
            ImGui::EndGroup();

            const float ruleX = origin.x + leftW;
            widgets::ruleAt(ImGui::GetWindowDrawList(), ImVec2(ruleX, origin.y), ImVec2(ruleX, origin.y + h), 2);

            const float rightX = ruleX + theme::crispPen(2) + px(14);
            ImGui::SetCursorScreenPos(ImVec2(rightX, origin.y));
            ImGui::BeginGroup();
            drawRight(ImVec2(std::max(px(100), origin.x + avail.x - rightX), h));
            ImGui::EndGroup();

            ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + h));
            ImGui::Dummy(ImVec2(avail.x, 0.0f));
            ImGui::PopStyleVar();
        }

    private:
        // Log lines come stamped "HH:MM:SS "; the plugin loader's start with "Plugin".
        void notePluginLog(const std::string& stamped) {
            std::string line = stamped;
            if (line.size() > 9 && line[2] == ':' && line[5] == ':' && line[8] == ' ') line.erase(0, 9);
            // "reloading through the worker…" says nothing about the outcome
            if (startsWith(line, "Plugin") && !startsWith(line, "Plugins: loading") && !startsWith(line, "Plugins: reloading"))
                lastLoad_ = line;
        }

        // --- the list ------------------------------------------------------------

        std::string userDir() const { return cleanDir(userPluginDirectory(false)); }

        int rowForPath(const std::string& path) const {
            for (std::size_t i = 0; i < rows_.size(); ++i)
                if (rows_[i].path == path) return static_cast<int>(i);
            return -1;
        }

        void refreshTree() {
            std::string keep = currentPath_;
            if (current_ >= 0 && current_ < static_cast<int>(rows_.size()) && rows_[static_cast<std::size_t>(current_)].isDir())
                keep = rows_[static_cast<std::size_t>(current_)].path;

            const Workbench& wb = app_.wb();
            // directories: the user folder first, then what the worker searched
            std::vector<std::string> dirs;
            auto addDir = [&dirs](const std::string& d) {
                if (d.empty()) return;
                if (std::find(dirs.begin(), dirs.end(), d) == dirs.end()) dirs.push_back(d);
            };
            const std::string user = userDir();
            addDir(user);
            for (const std::string& d : wb.pluginDirs()) addDir(cleanDir(d));

            // plugin files the worker reported, keyed by path
            std::map<std::string, Workbench::PluginInfo> known;
            for (const Workbench::PluginInfo& p : wb.plugins()) {
                const std::string file = cleanFile(p.file);
                known[file] = p;
                addDir(dirOf(file));
            }

            std::string home = cleanDir(platform::homeDirectory());
            rows_.clear();
            for (const std::string& dir : dirs) {
                std::error_code ec;
                const bool exists = fs::is_directory(fs::u8path(dir), ec);
                Row d;
                d.path = dir;
                d.text = !home.empty() && startsWith(dir, home) ? "~" + dir.substr(home.size()) : dir;
                d.tip = dir == user ? "Your plugins folder\n" + dir : dir;
                d.status = exists ? kDirectory : kMissingDirectory;
                rows_.push_back(d);

                // every .py on disk (helpers included) merged with the worker's view
                std::map<std::string, std::string> files;   // path -> file name
                if (exists) {
                    for (fs::directory_iterator it(fs::u8path(dir), ec), end; !ec && it != end; it.increment(ec)) {
                        std::error_code fe;
                        if (!it->is_regular_file(fe)) continue;
                        const std::string name = it->path().filename().u8string();
                        if (toLower(it->path().extension().u8string()) != ".py") continue;
                        files[cleanFile(it->path().u8string())] = name;
                    }
                }
                for (const auto& [path, info] : known)
                    if (dirOf(path) == dir) files[path] = fileName(path);

                std::vector<std::pair<std::string, std::string>> sorted(files.begin(), files.end());
                std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return lessNoCase(a.second, b.second); });
                for (const auto& [path, name] : sorted) {
                    Row r;
                    r.path = path;
                    auto it = known.find(path);
                    if (it == known.end()) {
                        const bool helper = startsWith(name, "_");
                        r.text = name;
                        r.status = kSkipped;
                        r.kind = helper ? "helper" : "not loaded";
                        r.tip = helper ? "Files starting with '_' are not loaded as operations" : "Not loaded yet: click Reload";
                        rows_.push_back(r);
                        continue;
                    }
                    const Workbench::PluginInfo& p = it->second;
                    r.text = p.name.empty() ? name : p.name;
                    r.kind = p.kind;
                    if (p.error.empty()) {
                        r.status = kLoaded;
                        r.tip = path + "\nLoaded as \"" + p.kind + "\"";
                    } else {
                        r.status = kFailed;
                        r.error = trimmed(p.error.substr(0, p.error.find('\n')));
                        r.tip = p.error;
                    }
                    rows_.push_back(r);
                }
            }
            current_ = keep.empty() ? -1 : rowForPath(keep);
        }

        std::string selectedDirectory() const {
            if (current_ < 0 || current_ >= static_cast<int>(rows_.size())) return userDir();
            const Row& r = rows_[static_cast<std::size_t>(current_)];
            return r.isDir() ? r.path : dirOf(r.path);
        }

        bool fileRowSelected() const {
            return current_ >= 0 && current_ < static_cast<int>(rows_.size()) && rows_[static_cast<std::size_t>(current_)].status >= kLoaded;
        }

        void selectRow(int index) {
            if (index < 0 || index >= static_cast<int>(rows_.size())) return;
            const Row& r = rows_[static_cast<std::size_t>(index)];
            if (r.isDir()) {   // a directory: keep the editor
                current_ = index;
                return;
            }
            const std::string path = r.path;
            if (path == currentPath_) {
                current_ = index;
                return;
            }
            // the selection stays where it was until the unsaved text is dealt with
            confirmDiscard([w = weak_from_this(), path] {
                if (auto self = w.lock()) {
                    self->current_ = self->rowForPath(path);
                    self->loadFile(path);
                }
            });
        }

        // --- the editor -----------------------------------------------------------

        void setModified(bool on) { modified_ = on; }

        std::string loadError() const {
            if (currentPath_.empty()) return {};
            std::string error;
            for (const Workbench::PluginInfo& p : app_.wb().plugins())
                if (cleanFile(p.file) == currentPath_ && !p.error.empty()) error = p.error;
            return error;
        }

        void showNothing() {
            currentPath_.clear();
            editor_.setText({});
            editor_.setReadOnly(true);
            readOnlyFile_ = false;
            setModified(false);
        }

        void loadFile(const std::string& path) {
            std::string text, error;
            if (!readWhole(path, text, error)) {
                app_.message("User operations", "Could not read " + path + ":\n" + error);
                return;
            }
            crlf_ = text.find("\r\n") != std::string::npos;
            if (crlf_) text = replaceAll(text, "\r\n", "\n");
            currentPath_ = path;
            editor_.setText(text);
#ifdef _WIN32
            // The permissions report the read-only attribute here (_waccess
            // checks no more than that either).
            std::error_code ec;
            const fs::perms perms = fs::status(fs::u8path(path), ec).permissions();
            const bool writable = !ec && (perms & fs::perms::owner_write) != fs::perms::none;
#else
            // Whether this user may write it: the owner's bit says nothing about
            // a file someone else owns, such as an installed example (root's).
            const bool writable = ::access(fs::u8path(path).c_str(), W_OK) == 0;
#endif
            editor_.setReadOnly(!writable);
            readOnlyFile_ = !writable;
            setModified(false);
        }

        // Writes the file and reloads the plugins (between two frames: the
        // worker may have to start first).
        bool save() {
            if (currentPath_.empty() || editor_.readOnly()) return false;
            std::string text = editor_.text();
            if (crlf_) text = replaceAll(text, "\n", "\r\n");
            std::ofstream out(fs::u8path(currentPath_), std::ios::binary | std::ios::trunc);
            if (!out || !(out << text) || !out.flush()) {
                std::error_code ec(errno, std::generic_category());
                app_.message("User operations", "Could not write " + currentPath_ + ":\n" + ec.message());
                return false;
            }
            out.close();
            setModified(false);
            reloadPlugins();
            return true;
        }

        void reloadPlugins(std::function<void()> then = {}) {
            app_.defer([w = weak_from_this(), then = std::move(then)] {
                auto self = w.lock();
                if (!self) return;
                self->app_.wb().loadPlugins(true);
                self->refreshTree();
                self->seenOperations_ = self->app_.bridge().rev().operations;
                if (then) then();
            });
        }

        void revert() {
            if (currentPath_.empty()) return;
            loadFile(currentPath_);
        }

        // Runs `proceed` once there is nothing unsaved in the way: at once,
        // or after the user saved or discarded. Cancel runs nothing.
        void confirmDiscard(std::function<void()> proceed) {
            if (!modified_ || currentPath_.empty()) {
                proceed();
                return;
            }
            app_.ask("User operations", fileName(currentPath_) + " has unsaved changes.\n\nSave them before switching?",
                     {"Cancel", "Discard", "Save"}, [w = weak_from_this(), proceed = std::move(proceed)](int button) {
                         auto self = w.lock();
                         if (!self) return;
                         if (button == 2) {
                             if (self->save()) proceed();
                         } else if (button == 1) {
                             // Back to the file on disk: the discarded text would
                             // otherwise stay in the editor, unmarked, for a later
                             // Save to write. A file that can no longer be read
                             // leaves the editor empty.
                             self->revert();
                             if (self->modified_) self->showNothing();
                             proceed();
                         }
                     });
        }

        // --- actions -------------------------------------------------------------------

        void newPlugin() {
            confirmDiscard([w = weak_from_this()] {
                auto self = w.lock();
                if (!self) return;
                self->app_.promptText("New user operation", "Name (becomes <name>.py in your plugins folder):", "my_filter",
                                      [w](const std::string& raw) {
                                          if (auto s = w.lock()) s->createPlugin(raw);
                                      });
            });
        }

        void createPlugin(const std::string& raw) {
            const std::string words = identifier(raw);
            if (words.empty()) {
                app_.message("New user operation", "The name needs at least one letter or digit.");
                return;
            }
            // A kind may not start with a digit, and the loader skips a file
            // whose name starts with '_' as a helper: such a name gets a
            // prefix of letters, which the title leaves out.
            const std::string kind = std::isdigit(static_cast<unsigned char>(words[0])) ? "op_" + words : words;
            const std::string dir = cleanDir(userPluginDirectory(true));
            std::error_code ec;
            if (!fs::is_directory(fs::u8path(dir), ec)) {
                app_.message("New user operation", "Could not create the plugins folder " + dir + ".");
                return;
            }
            const std::string path = cleanFile(dir + "/" + kind + ".py");
            auto open = [w = weak_from_this(), path] {
                if (auto self = w.lock()) self->openFile(path);
            };
            if (fs::exists(fs::u8path(path), ec)) {
                app_.message("New user operation", path + " already exists; opening it instead.", MessageIcon::Info);
                reloadPlugins(open);
                return;
            }
            std::ofstream out(fs::u8path(path), std::ios::binary);
            if (!out || !(out << pluginTemplate(kind, titleCase(words))) || !out.flush()) {
                std::error_code we(errno, std::generic_category());
                app_.message("New user operation", "Could not write " + path + ":\n" + we.message());
                return;
            }
            out.close();
            refreshTree();
            open();   // shown at once, while the worker loads it
            reloadPlugins(open);
        }

        void openFolder() {
            std::string dir = selectedDirectory();
            std::error_code ec;
            if (!fs::is_directory(fs::u8path(dir), ec)) {
                if (dir == userDir()) dir = cleanDir(userPluginDirectory(true));
                if (!fs::is_directory(fs::u8path(dir), ec)) {
                    app_.message("User operations", dir + " does not exist.", MessageIcon::Info);
                    return;
                }
                refreshTree();
            }
            platform::openInFileManager(dir);
        }

        void deleteCurrent() {
            if (!fileRowSelected()) return;
            const Row r = rows_[static_cast<std::size_t>(current_)];
            app_.ask("Delete user operation", "Delete " + r.text + "?\n\nThis removes the file\n" + r.path, {"Yes", "Cancel"},
                     [w = weak_from_this(), path = r.path](int button) {
                         auto self = w.lock();
                         if (!self || button != 0) return;
                         std::error_code ec;
                         if (!fs::remove(fs::u8path(path), ec) || ec) {
                             self->app_.message("Delete user operation", "Could not delete " + path + ".");
                             return;
                         }
                         if (path == self->currentPath_) self->showNothing();
                         self->refreshTree();
                         self->reloadPlugins();
                     });
        }

        // --- drawing -------------------------------------------------------------------

        void drawRow(std::size_t i, float width) {
            const Row& r = rows_[i];
            const bool selected = static_cast<int>(i) == current_;
            const float h = r.isDir() ? kDirRowHeight : (r.status == kFailed ? kErrorRowHeight : kFileRowHeight);
            ImGui::PushID(r.path.c_str());
            widgets::RowOpts o;
            o.selected = selected;
            o.edge = !r.isDir();
            o.topRule = r.isDir() ? 1.0f : 0.0f;
            o.width = width / theme::scale();
            const widgets::Row row = widgets::beginRow("##row", h, o);
            if (!r.tip.empty()) widgets::tooltip(r.tip);
            if (!scrollTo_.empty() && scrollTo_ == r.path) {
                ImGui::SetScrollHereY(0.5f);
                scrollTo_.clear();
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 min = row.min, max = row.max;
            if (r.isDir()) {
                ImFont* f = theme::captionFont();
                std::string text = r.text;
                if (r.status == kMissingDirectory) text += "  (not created)";
                const float room = std::max(0.0f, max.x - min.x - px(20));
                const std::string shown = elideMiddle(captionCase(text), room, f, theme::kCaptionPx);
                const float lh = lineHeight(f, theme::kCaptionPx);
                drawIn(dl, f, theme::kCaptionPx, ImVec2(min.x + px(10), min.y + (px(kDirRowHeight) - lh) * 0.5f),
                       r.status == kMissingDirectory ? theme::kNeutral400 : theme::kNeutral600, shown);
            } else {
                // status glyph at the right
                const float lineH = px(kFileRowHeight);
                const ImVec2 gMin(max.x - px(26), min.y), gMax(max.x - px(6), min.y + lineH);
                if (r.status == kLoaded)
                    drawIcon(dl, ImVec2(gMin.x + px(4), gMin.y + px(4)), ImVec2(gMax.x - px(4), gMax.y - px(4)), Icon::Check,
                             theme::kNeutral600, px(1.5f));
                else if (r.status == kFailed)
                    drawIcon(dl, ImVec2(gMin.x + px(4), gMin.y + px(4)), ImVec2(gMax.x - px(4), gMax.y - px(4)), Icon::Close,
                             theme::kAccent, px(1.5f));
                // name, then the kind in caption style
                ImFont* nameFont = theme::font(Weight::ExtraBold);
                ImFont* kindFont = theme::captionFont();
                const std::string kind = captionCase(r.kind);
                const float left = min.x + px(14), right = gMin.x - px(6);
                const float kindW = kind.empty() ? 0.0f : measure(kindFont, theme::kCaptionPx, kind) + px(8);
                const std::string name = elideRight(r.text, std::max(px(20), right - left - kindW), nameFont, 12);
                const float nameH = lineHeight(nameFont, 12);
                drawIn(dl, nameFont, 12, ImVec2(left, min.y + (lineH - nameH) * 0.5f), theme::kText, name);
                if (!kind.empty()) {
                    const float nameW = measure(nameFont, 12, name);
                    const float room = std::max(0.0f, right - left - nameW - px(8));
                    const std::string k = elideRight(kind, room, kindFont, theme::kCaptionPx);
                    const float kh = lineHeight(kindFont, theme::kCaptionPx);
                    drawIn(dl, kindFont, theme::kCaptionPx, ImVec2(left + nameW + px(8), min.y + px(1) + (lineH - kh) * 0.5f),
                           theme::kNeutral600, k);
                }
                if (r.status == kFailed) {
                    ImFont* ef = theme::font(Weight::Regular);
                    const std::string e = elideRight(r.error, std::max(0.0f, max.x - px(10) - left), ef, theme::kSmallPx);
                    drawIn(dl, ef, theme::kSmallPx, ImVec2(left, min.y + lineH - px(6)), theme::kAccentText, e);
                }
            }
            if (row.clicked) selectRow(static_cast<int>(i));
            widgets::endRow(row);
            ImGui::PopID();
        }

        void drawLeft(ImVec2 size) {
            const float buttonH = theme::snap(std::max(px(14), theme::textSize("Ag", 12).y) + px(8) + 2 * px(theme::kBorder));
            const bool hasStatus = !lastLoad_.empty();
            const float statusH = hasStatus ? 2 * theme::textSize("Ag", theme::kSmallPx).y + px(8) : 0.0f;
            const float treeH = std::max(px(60), size.y - buttonH - px(8) - statusH);

            ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kTransparent);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
            if (ImGui::BeginChild("##pluginTree", ImVec2(size.x, treeH), ImGuiChildFlags_None)) {
                const float w = ImGui::GetContentRegionAvail().x;
                for (std::size_t i = 0; i < rows_.size(); ++i) drawRow(i, w);
                // arrows move through the list while it has the keyboard
                if (ImGui::IsWindowFocused() && !rows_.empty() && !ImGui::IsAnyItemActive()) {
                    const int n = static_cast<int>(rows_.size());
                    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
                        const int next = std::min(n - 1, current_ + 1);
                        selectRow(next);
                        scrollTo_ = rows_[static_cast<std::size_t>(next)].path;
                    } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
                        const int prev = std::max(0, current_ - 1);
                        selectRow(prev);
                        scrollTo_ = rows_[static_cast<std::size_t>(prev)].path;
                    }
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();

            ImGui::Dummy(ImVec2(0.0f, px(8)));
            auto small = [](const char* label, const char* tip, bool enabled = true) {
                widgets::ButtonOpts o;
                o.small = true;
                o.tooltip = tip;
                o.enabled = enabled;
                return widgets::button(label, o);
            };
            const float rowX = ImGui::GetCursorPosX();
            if (small("New", "Create a plugin from a template in your plugins folder")) newPlugin();
            ImGui::SameLine(0.0f, px(6));
            if (small("Open folder", "Show the selected folder in the file manager")) openFolder();
            ImGui::SameLine(0.0f, px(6));
            if (small("Delete", "Delete the selected plugin file", fileRowSelected())) deleteCurrent();
            ImGui::SameLine(0.0f, 0.0f);
            {
                const float reloadW = theme::textSize("Reload", 12, Weight::SemiBold).x + 2 * px(10) + 2 * px(theme::kBorder);
                const float x = std::max(ImGui::GetCursorPosX() + px(6), rowX + size.x - reloadW);
                ImGui::SetCursorPosX(x);
            }
            if (small("Reload", "Re-import every plugin")) reloadPlugins();

            if (hasStatus) {
                // what the last load said, so a worker that cannot start is not silent
                ImGui::Dummy(ImVec2(0.0f, px(6)));
                const bool bad = startsWith(lastLoad_, "Plugins unavailable") || startsWith(lastLoad_, "Plugin error");
                const std::string one = simplified(lastLoad_);
                ImFont* f = theme::font(Weight::Regular);
                const float lh = theme::textSize("Ag", theme::kSmallPx).y;
                const ImVec2 at = ImGui::GetCursorScreenPos();
                // two lines at most: the first line, then the rest elided
                std::string first = one, second;
                if (measure(f, theme::kSmallPx, one) > size.x) {
                    std::size_t cut = 0;
                    for (std::size_t i = 0; i < one.size(); ++i)
                        if (one[i] == ' ' && measure(f, theme::kSmallPx, one.substr(0, i)) <= size.x) cut = i;
                    if (cut == 0) cut = std::min<std::size_t>(one.size(), 40);
                    first = one.substr(0, cut);
                    second = elideRight(trimmed(one.substr(cut)), size.x, f, theme::kSmallPx);
                }
                const ImU32 c = bad ? theme::kAccentText : theme::kNeutral600;
                drawIn(ImGui::GetWindowDrawList(), f, theme::kSmallPx, at, c, first);
                if (!second.empty()) drawIn(ImGui::GetWindowDrawList(), f, theme::kSmallPx, ImVec2(at.x, at.y + lh), c, second);
                ImGui::InvisibleButton("##loadStatus", ImVec2(size.x, 2 * lh));
                widgets::tooltip(lastLoad_);
            }
        }

        void drawRight(ImVec2 size) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            float y = origin.y;

            // path + "modified"
            {
                ImFont* mono = theme::mono();
                const float lh = std::max(lineHeight(mono, 11), theme::textSize("Ag", theme::kCaptionPx).y);
                float right = origin.x + size.x;
                if (modified_) {
                    const std::string mark = captionCase("modified");
                    const float mw = measure(theme::captionFont(), theme::kCaptionPx, mark);
                    drawIn(dl, theme::captionFont(), theme::kCaptionPx, ImVec2(right - mw, y + (lh - lineHeight(theme::captionFont(), theme::kCaptionPx)) * 0.5f),
                           theme::kAccentText, mark);
                    right -= mw + px(10);
                }
                const std::string full = currentPath_.empty() ? std::string("No file selected") : currentPath_;
                const std::string shown = elideMiddle(full, std::max(0.0f, right - origin.x), mono, 11);
                drawIn(dl, mono, 11, ImVec2(origin.x, y + (lh - lineHeight(mono, 11)) * 0.5f), theme::kNeutral700, shown);
                ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
                ImGui::InvisibleButton("##path", ImVec2(std::max(1.0f, right - origin.x), lh));
                if (shown != full) widgets::tooltip(full);
                y += lh + px(8);
            }
            if (readOnlyFile_ && !currentPath_.empty()) {
                ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
                widgets::textWrapped("Read-only: this file is not writable. Copy it into your plugins folder to change it.",
                                     theme::kSmallPx, theme::kNeutral600, Weight::Regular, size.x);
                y = ImGui::GetCursorScreenPos().y + px(8);
            }

            // the banner of a file that did not load
            const std::string error = loadError();
            float bannerH = 0.0f, textH = 0.0f;
            if (!error.empty()) {
                int lines = 1;
                for (char c : error)
                    if (c == '\n') ++lines;
                if (!error.empty() && error.back() == '\n') --lines;
                lines = std::clamp(lines, 1, 6);
                textH = static_cast<float>(lines) * lineHeight(theme::mono(), 11) + px(10);
                bannerH = px(8) + theme::textSize("Ag", theme::kCaptionPx).y + px(4) + textH + px(8);
            }
            const float buttonH = theme::snap(std::max(px(18), theme::textSize("Ag", 13).y) + 2 * px(7) + 2 * px(theme::kBorder));
            const float rule = theme::crispPen(2);
            const float bottom = origin.y + size.y;
            const float editorH =
                std::max(px(80), bottom - y - 2 * rule - px(8) - (bannerH > 0.0f ? bannerH + px(8) : 0.0f) - buttonH);

            // editor between two rules
            dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(origin.x + size.x, y + rule), theme::kDivider);
            y += rule;
            ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
            if (editor_.draw("##pluginEditor", ImVec2(size.x, editorH)) && !currentPath_.empty() && !modified_) setModified(true);
            y += editorH;
            dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(origin.x + size.x, y + rule), theme::kDivider);
            y += rule + px(8);

            if (!error.empty()) {
                const ImVec2 bMin(origin.x, y), bMax(origin.x + size.x, y + bannerH);
                dl->AddRectFilled(bMin, bMax, theme::kSurface);
                dl->AddRectFilled(bMin, ImVec2(bMin.x + theme::snap(px(3)), bMax.y), theme::kAccent);
                const float x = bMin.x + theme::snap(px(3)) + px(10);
                ImGui::SetCursorScreenPos(ImVec2(x, y + px(8)));
                widgets::caption("Did not load", theme::kAccentText);
                ImGui::SetCursorScreenPos(ImVec2(x, y + px(8) + theme::textSize("Ag", theme::kCaptionPx).y + px(4)));
                {
                    // selectable, so a traceback can be copied
                    const theme::FontScope f(11, theme::mono());
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, theme::kTransparent);
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccent700);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, px(2)));
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                    bannerText_ = error;
                    ImGui::InputTextMultiline("##loadError", &bannerText_, ImVec2(std::max(px(40), bMax.x - px(10) - x), textH),
                                              ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(2);
                }
                y += bannerH + px(8);
            }

            // Save · Revert · hint ……… Close
            ImGui::SetCursorScreenPos(ImVec2(origin.x, std::max(y, bottom - buttonH)));
            {
                widgets::ButtonOpts o;
                o.kind = widgets::ButtonKind::Primary;
                o.enabled = !currentPath_.empty() && !editor_.readOnly() && modified_;
                o.tooltip = "Write the file and reload plugins (Ctrl+S)";
                if (widgets::button("Save", o)) save();
            }
            ImGui::SameLine(0.0f, px(8));
            {
                widgets::ButtonOpts o;
                o.enabled = !currentPath_.empty();   // also re-reads a file changed outside
                o.tooltip = "Discard the changes and reload the file from disk";
                if (widgets::button("Revert", o)) revert();
            }
            ImGui::SameLine(0.0f, px(16));
            {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float th = theme::textSize("Ag", theme::kSmallPx).y;
                ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + (buttonH - th) * 0.5f));
                widgets::text("Tab indents \xC2\xB7 Ctrl+/ comments \xC2\xB7 Ctrl+S saves", theme::kSmallPx, theme::kNeutral500);
                ImGui::SameLine(0.0f, 0.0f);
                ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, p.y));
            }
            {
                const float closeW = theme::textSize("Close", 13, Weight::SemiBold).x + 2 * px(8) + 2 * px(theme::kBorder);
                const float x = std::max(ImGui::GetCursorScreenPos().x + px(8), origin.x + size.x - closeW);
                ImGui::SetCursorScreenPos(ImVec2(x, ImGui::GetCursorScreenPos().y));
                widgets::ButtonOpts o;
                o.kind = widgets::ButtonKind::Ghost;
                if (widgets::button("Close", o)) {
                    confirmDiscard([w = weak_from_this()] {
                        if (auto self = w.lock()) self->close();
                    });
                }
            }
        }

        App& app_;
        widgets::CodeEditor editor_;
        std::vector<Row> rows_;
        int current_ = -1;
        std::string currentPath_;   // file shown in the editor, empty when none
        bool modified_ = false;
        bool readOnlyFile_ = false;
        bool crlf_ = false;          // the file's own line endings, kept on save
        bool refreshOnShow_ = false;
        std::string scrollTo_;
        std::string lastLoad_;
        std::string bannerText_;
        std::uint64_t seenOperations_ = 0;
        int logConnection_ = 0;
    };

    std::shared_ptr<PluginManager> makePluginManager(App& app) { return std::make_shared<PluginManagerDialog>(app); }

} // namespace sirius::app::gui
