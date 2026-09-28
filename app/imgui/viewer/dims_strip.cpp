#include "imgui/viewer/dims_strip.hpp"

#include <algorithm>
#include <cstdint>

#include <imgui_internal.h>

#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {
        // The grid of the strip: 14 px margins left and right, 8 above and
        // below, columns 120 | 1fr | 80 with 14 px between, rows 6 px apart.
        constexpr float kMarginX = 14, kMarginY = 8, kCol0 = 120, kCol2 = 80, kGap = 14, kRowGap = 6, kRowH = 20;

        // A slider's extra keys: page up / down step a tenth of the axis,
        // home / end go to its ends (the arrows are the slider's own). The
        // slider owns them while it has focus, so a shortcut on the same key
        // stands back, as it does for the arrows.
        bool pageKeys(std::int64_t* v, std::int64_t n) {
            if (!ImGui::IsItemFocused() || n <= 1) return false;
            for (ImGuiKey key : {ImGuiKey_PageUp, ImGuiKey_PageDown, ImGuiKey_Home, ImGuiKey_End})
                ImGui::SetKeyOwner(key, ImGui::GetItemID());
            const std::int64_t page = std::max<std::int64_t>(1, n / 10);
            std::int64_t nv = *v;
            if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) nv -= page;
            if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) nv += page;
            if (ImGui::IsKeyPressed(ImGuiKey_Home)) nv = 0;
            if (ImGui::IsKeyPressed(ImGuiKey_End)) nv = n - 1;
            nv = std::clamp<std::int64_t>(nv, 0, n - 1);
            if (nv == *v) return false;
            *v = nv;
            return true;
        }
    } // namespace

    void DimsStrip::setExtents(Index nz, Index nt, double dzUm, double frameIntervalS) {
        nz_ = std::max<Index>(nz, 1);
        nt_ = std::max<Index>(nt, 1);
        dz_ = dzUm;
        dt_ = frameIntervalS;
        setPosition(z_, t_);
    }

    void DimsStrip::setPosition(Index z, Index t) {
        z_ = std::clamp<Index>(z, 0, nz_ - 1);
        t_ = std::clamp<Index>(t, 0, nt_ - 1);
    }

    float DimsStrip::height() const {
        const float rows = nt_ > 1 ? 2.0f * kRowH + kRowGap : kRowH;
        return theme::snap(px(2.0f * kMarginY + rows));
    }

    void DimsStrip::draw(ImVec2 min, ImVec2 max) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImGui::PushID("dims");
        const float left = min.x + px(kMarginX), right = max.x - px(kMarginX);
        const float sliderX0 = left + px(kCol0) + px(kGap);
        const float sliderX1 = std::max(sliderX0 + px(40), right - px(kCol2) - px(kGap));
        const float sliderW = (sliderX1 - sliderX0) / std::max(theme::scale(), 0.01f);
        const float rowH = px(kRowH);

        auto row = [&](float top, const char* axis, std::int64_t value, std::int64_t n, const std::string& readout, bool withPlay,
                       const std::function<void(Index)>& request) {
            const float cy = top + rowH * 0.5f;
            // head: axis letter (12 / 800), play button, readout (12, neutral-600)
            float x = left;
            const ImVec2 axisSize = theme::textSize(axis, 12, theme::Weight::ExtraBold);
            widgets::drawText(dl, ImVec2(x, cy - axisSize.y * 0.5f), axis, 12, theme::kText, theme::Weight::ExtraBold);
            x += axisSize.x + px(10);
            if (withPlay) {
                ImGui::SetCursorScreenPos(ImVec2(x, theme::snap(cy - px(10))));
                widgets::GlyphOpts o;
                o.border = theme::kText;
                o.tooltip = "Play / pause the time series (space in the viewer)";
                if (widgets::glyphButton("##play", playing_ ? Icon::Pause : Icon::Play, 20, o) && playToggled) playToggled(!playing_);
                x += px(20) + px(10);
            }
            const ImVec2 rs = theme::textSize(readout, 12);
            widgets::drawText(dl, ImVec2(x, cy - rs.y * 0.5f), readout, 12, theme::kNeutral600);

            // the slider
            ImGui::SetCursorScreenPos(ImVec2(sliderX0, theme::snap(cy - px(9))));
            widgets::SliderOpts so;
            so.width = sliderW;
            so.enabled = n > 1;
            std::int64_t v = value;
            bool changed = widgets::sliderInt(axis, &v, 0, std::max<std::int64_t>(n - 1, 0), so);
            changed = pageKeys(&v, n) || changed;
            if (changed && v != value && request) request(static_cast<Index>(v));

            // "n / max", right-aligned
            const std::string pos = format("%lld / %lld", static_cast<long long>(value), static_cast<long long>(n - 1));
            const ImVec2 ps = theme::textSize(pos, 12);
            widgets::drawText(dl, ImVec2(right - ps.x, cy - ps.y * 0.5f), pos, 12, theme::kText);
        };

        const float top = min.y + px(kMarginY);
        row(top, "Z", static_cast<std::int64_t>(z_), static_cast<std::int64_t>(nz_), format("%.2f \xC2\xB5m", static_cast<double>(z_) * dz_), false,
            zRequested);
        if (nt_ > 1) {
            const std::string sec = dt_ > 0.0 ? format("%.1f s", static_cast<double>(t_) * dt_) : format("frame %lld", static_cast<long long>(t_));
            row(top + rowH + px(kRowGap), "T", static_cast<std::int64_t>(t_), static_cast<std::int64_t>(nt_), sec, true, tRequested);
        }
        ImGui::PopID();
    }

} // namespace sirius::app::gui
