#ifndef SIRIUS_IMGUI_DIALOGS_OPEN_DATASET_COMMON_HPP
#define SIRIUS_IMGUI_DIALOGS_OPEN_DATASET_COMMON_HPP

// What the Open dataset and the Open folder dialogs share: the thread that
// does their file system work (a probe, a directory listing, a pattern
// matched against thousands of names), the rows both lay out by hand (a
// field and the buttons beside it, the raw SIM layout, the voxel sizes) and
// the button row at the bottom.
//
// Threads: a dialog owns one Worker. Its jobs run in the order they were
// given and report with Bridge::post; what they post checks the dialog's
// `alive` flag first, which the destructor clears before it joins the
// thread, so a result that arrives late finds nothing to touch.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <imgui.h>

#include "imgui/app.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui::dataset_dialogs {

    // The folder dialog as the Open dialog's Folder… raises it: `opened` is
    // called when the dataset has started to open, so the Open dialog under
    // it closes too (and remembers the folder), as the Qt dialog did after
    // its nested exec() came back accepted. (folder_dataset_dialog.cpp)
    std::shared_ptr<Dialog> makeFolderDatasetDialog(App& app, const std::string& folder, std::function<void()> opened);

    // --- paths ---------------------------------------------------------------
    // The GUI's strings are UTF-8; std::filesystem takes them as such only
    // when told so.
    inline std::filesystem::path toPath(const std::string& utf8) { return std::filesystem::u8path(utf8); }

    inline std::string fromPath(const std::filesystem::path& p) {
        const auto u = p.u8string();
        return std::string(u.begin(), u.end());
    }

    inline std::filesystem::path canonicalPath(const std::filesystem::path& p) {
        std::error_code ec;
        const std::filesystem::path c = std::filesystem::weakly_canonical(p, ec);
        if (!ec) return c;
        const std::filesystem::path a = std::filesystem::absolute(p, ec);
        return ec ? p : a;
    }

    // --- the dialog's own thread ------------------------------------------------
    class Worker {
    public:
        Worker() : thread_([this] { loop(); }) {}
        ~Worker() {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                quit_ = true;
                jobs_.clear();   // what has not started is not wanted any more
            }
            ready_.notify_all();
            if (thread_.joinable()) thread_.join();
        }
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;

        void run(std::function<void()> job) {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                jobs_.push_back(std::move(job));
            }
            ready_.notify_one();
        }

    private:
        void loop() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [this] { return quit_ || !jobs_.empty(); });
                    if (quit_) return;
                    job = std::move(jobs_.front());
                    jobs_.pop_front();
                }
                try {
                    job();
                } catch (const std::exception&) {
                    // a job reports its own errors; one that threw past that has nobody to tell
                }
            }
        }

        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<std::function<void()>> jobs_;
        bool quit_ = false;
        std::thread thread_;   // last: it uses the members above
    };

    using Alive = std::shared_ptr<std::atomic<bool>>;
    inline Alive makeAlive() { return std::make_shared<std::atomic<bool>>(true); }

    // --- layout ----------------------------------------------------------------
    // The next item `designPx` below the last one, whatever the item spacing is.
    inline void gap(float designPx) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y + theme::px(designPx));
    }

    // The size widgets::button gives a label.
    inline ImVec2 buttonSize(const std::string& label, widgets::ButtonKind kind, bool small) {
        const bool ghost = kind == widgets::ButtonKind::Ghost;
        const float fontSize = small ? 12.0f : 13.0f;
        const float padX = small ? (ghost ? 8.0f : 10.0f) : (ghost ? 8.0f : 12.0f);
        const float padY = small ? 4.0f : 7.0f;
        const float minH = small ? 14.0f : 18.0f;
        const theme::Weight weight = kind == widgets::ButtonKind::Primary ? theme::Weight::ExtraBold : theme::Weight::SemiBold;
        const ImVec2 ts = theme::textSize(label, fontSize, weight);
        return ImVec2(theme::snap(ts.x + 2 * theme::px(padX) + 2 * theme::px(theme::kBorder)),
                      theme::snap(std::max(theme::px(minH), ts.y) + 2 * theme::px(padY) + 2 * theme::px(theme::kBorder)));
    }

    // The size of a widgets::checkbox with this label.
    inline ImVec2 checkSize(const std::string& label) {
        const ImVec2 ts = theme::textSize(label, 12);
        return ImVec2(theme::snap(theme::px(14)) + theme::px(8) + ts.x, std::max(theme::snap(theme::px(14)), ts.y) + theme::px(4));
    }

    // One line of controls of different heights, each centred on the line:
    // a field and the small buttons beside it, a checkbox and two spin boxes.
    class Line {
    public:
        explicit Line(float heightDesign = theme::kInputH)
            : origin_(ImGui::GetCursorScreenPos()), width_(std::max(1.0f, ImGui::GetContentRegionAvail().x)),
              height_(theme::snap(theme::px(heightDesign))) {}

        float width() const noexcept { return width_; }
        float height() const noexcept { return height_; }
        // The next item, `itemHeight` display pixels high, starts `x` display
        // pixels from the left of the line.
        void at(float x, float itemHeight) const {
            ImGui::SetCursorScreenPos(ImVec2(theme::snap(origin_.x + x), origin_.y + std::floor((height_ - itemHeight) * 0.5f)));
        }
        void text(float x, const std::string& s, float px, ImU32 color) const {
            at(x, theme::textSize(s, px).y);
            widgets::text(s, px, color);
        }
        void end() const {
            ImGui::SetCursorScreenPos(origin_);
            ImGui::Dummy(ImVec2(width_, height_));
        }

    private:
        ImVec2 origin_;
        float width_, height_;
    };

    // Columns of labelled fields that start on one line, laid out by position
    // rather than with SameLine: groups side by side would align the labels on
    // the fields' text baseline, and a spin box ends by moving the cursor
    // back, which a group must not end on. end() submits the one item that
    // covers them all.
    class Columns {
    public:
        Columns() : origin_(ImGui::GetCursorScreenPos()), width_(std::max(1.0f, ImGui::GetContentRegionAvail().x)), maxY_(origin_.y) {}

        float width() const noexcept { return width_; }
        // The next column starts `x` display pixels from the left.
        void at(float x) const { ImGui::SetCursorScreenPos(ImVec2(theme::snap(origin_.x + x), origin_.y)); }
        // at(x), the 11 px label, and the cursor back at x for the field under it.
        void labelled(float x, const std::string& label) const {
            at(x);
            widgets::fieldLabel(label);
            ImGui::SetCursorScreenPos(ImVec2(theme::snap(origin_.x + x), ImGui::GetCursorScreenPos().y));
        }
        // After the last item of a column.
        void track() { maxY_ = std::max(maxY_, ImGui::GetItemRectMax().y); }
        void end() const {
            ImGui::SetCursorScreenPos(origin_);
            ImGui::Dummy(ImVec2(width_, std::max(1.0f, maxY_ - origin_.y)));
        }

    private:
        ImVec2 origin_;
        float width_;
        float maxY_;
    };

    // A number field that says its unit inside, as a spin box with a suffix does.
    inline bool unitField(const char* id, double* value, double lo, double hi, int decimals, const char* unit, float widthDisplay,
                          bool enabled = true) {
        widgets::FieldOpts f;
        f.width = widthDisplay / std::max(theme::scale(), 0.01f);
        f.enabled = enabled;
        const bool changed = widgets::inputDouble(id, value, lo, hi, 0.0, decimals, f);
        const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
        widgets::drawTextIn(ImGui::GetWindowDrawList(), min, ImVec2(max.x - theme::px(8), max.y), unit, 12,
                            enabled ? theme::kNeutral600 : theme::kNeutral400, theme::Weight::Regular, 1.0f, 0.5f);
        return changed;
    }

    // "Voxel x | Voxel y | Voxel z" in the first three columns, each `column`
    // display pixels wide and `spacing` apart.
    inline bool voxelFields(Columns& cols, double* vx, double* vy, double* vz, float column, float spacing) {
        bool changed = false;
        const char* labels[3] = {"Voxel x", "Voxel y", "Voxel z"};
        const char* ids[3] = {"##vx", "##vy", "##vz"};
        double* values[3] = {vx, vy, vz};
        for (int i = 0; i < 3; ++i) {
            cols.labelled(static_cast<float>(i) * (column + spacing), labels[i]);
            changed = unitField(ids[i], values[i], 0.0001, 1000.0, 4, "µm", column) || changed;
            cols.track();
        }
        return changed;
    }

    // "[ ] Raw SIM acquisition ...            dirs [3]  phases [5]  [ ] fast SI order"
    struct SimFields {
        bool present = false;
        std::int64_t dirs = 3;
        std::int64_t phases = 5;
        bool fastSi = false;
    };

    inline void simRow(SimFields& sim) {
        const float scale = std::max(theme::scale(), 0.01f);
        const Line line;
        const std::string fast = "fast SI order";
        const float spin = theme::px(62), space = theme::px(10), labelGap = theme::px(6);
        const float dirsLabel = theme::textSize("dirs", 12).x, phasesLabel = theme::textSize("phases", 12).x;
        const ImVec2 fastSize = checkSize(fast);
        float x = line.width() - fastSize.x;
        const float fastX = x;
        x -= space + spin;
        const float phasesX = x;
        x -= labelGap + phasesLabel;
        const float phasesLabelX = x;
        x -= space + spin;
        const float dirsX = x;
        x -= labelGap + dirsLabel;
        const float dirsLabelX = x;

        const std::string label = "Raw SIM acquisition: z holds directions × phases × planes";
        line.at(0.0f, checkSize(label).y);
        widgets::checkbox(label.c_str(), &sim.present);
        const ImU32 ink = sim.present ? theme::kNeutral700 : theme::kNeutral400;
        widgets::FieldOpts f;
        f.width = spin / scale;
        f.enabled = sim.present;
        line.text(dirsLabelX, "dirs", 12, ink);
        line.at(dirsX, line.height());
        widgets::inputInt("##dirs", &sim.dirs, 1, 9, 1, f);
        line.text(phasesLabelX, "phases", 12, ink);
        line.at(phasesX, line.height());
        widgets::inputInt("##phases", &sim.phases, 1, 15, 1, f);
        line.at(fastX, fastSize.y);
        widgets::checkbox(fast.c_str(), &sim.fastSi, sim.present);
        line.end();
    }

    // widgets::tooltip for an item that may be disabled: a Qt widget keeps its
    // tool tip when it is disabled, and says why it is.
    inline void tooltipEvenDisabled(const std::string& s) {
        if (s.empty() || !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) return;
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, theme::px(8, 4));
        if (ImGui::BeginTooltip()) {
            {
                // popped before the tooltip ends: Dear ImGui checks the font stack at End()
                const theme::FontScope f(12);
                ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
                ImGui::PushTextWrapPos(theme::px(360));
                ImGui::TextUnformatted(s.c_str(), s.c_str() + s.size());
                ImGui::PopTextWrapPos();
                ImGui::PopStyleColor();
            }
            ImGui::EndTooltip();
        }
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }

    // --- the button row ----------------------------------------------------------
    struct FooterButton {
        std::string label;
        widgets::ButtonKind kind = widgets::ButtonKind::Secondary;
        bool enabled = true;
        std::string tooltip;
    };

    // Flush right, in the order given (the primary button last). Returns the
    // index of the button pressed, or -1.
    inline int footer(const std::vector<FooterButton>& buttons) {
        const float spacing = theme::px(8);
        std::vector<float> widths;
        float total = 0.0f;
        for (std::size_t i = 0; i < buttons.size(); ++i) {
            const float w = std::max(theme::px(84), buttonSize(buttons[i].label, buttons[i].kind, false).x);
            widths.push_back(w);
            total += w + (i ? spacing : 0.0f);
        }
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - total));
        int pressed = -1;
        for (std::size_t i = 0; i < buttons.size(); ++i) {
            if (i) ImGui::SameLine(0.0f, spacing);
            widgets::ButtonOpts o;
            o.kind = buttons[i].kind;
            o.width = widths[i] / std::max(theme::scale(), 0.01f);
            o.centered = true;
            o.enabled = buttons[i].enabled;
            o.tooltip = buttons[i].tooltip;
            if (widgets::button((buttons[i].label + "##footer" + std::to_string(i)).c_str(), o)) pressed = static_cast<int>(i);
        }
        return pressed;
    }

    // Whether a popup (a dropdown, a menu, a message box) is open above the
    // dialog being drawn. Asked at the start of draw(): Enter that picks an
    // item in a dropdown closes it within the frame.
    inline bool popupAbove() { return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId); }

    // Enter, as the default button of a dialog takes it: only in the dialog
    // that has the keyboard, and not when it went to a popup of the dialog
    // (`popupAtStart`: popupAbove() at the start of the frame) or to a field
    // that takes Enter for itself (`consumed`).
    inline bool enterPressed(bool popupAtStart, bool consumed = false) {
        return !popupAtStart && !consumed && !popupAbove() && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
               (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false));
    }

} // namespace sirius::app::gui::dataset_dialogs

#endif // SIRIUS_IMGUI_DIALOGS_OPEN_DATASET_COMMON_HPP
