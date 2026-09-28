#include "imgui/panels/track_table.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <imgui.h>

#include "core/labels.hpp"   // labelColor
#include "imgui/panels/diagnostic_cells.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {

        const char* const kNames[TrackTable::ColumnCount] = {"ID", "FRAMES", "PRESENT", "GAPS", "µm / FRAME", "NET µm", "PARENT", "CHILDREN"};
        const char* const kTips[TrackTable::ColumnCount] = {
            "Track id: the label id it has in every frame",
            "First and last frame the track is present in",
            "Frames the track is present in",
            "Frames missing between its first and last: where it was lost and picked up again",
            "Centroid path length divided by the frames it spans",
            "Distance from the first centroid to the last",
            "The track it divided from, when the tracker reported one",
            "The tracks it divided into",
        };

        const char* const kDash = "—";

        std::string cellText(const TrackSummary& s, int column) {
            switch (column) {
                case TrackTable::Id: return format("%04u", static_cast<unsigned>(s.id));
                case TrackTable::Frames: return format("%lld – %lld", static_cast<long long>(s.first), static_cast<long long>(s.last));
                case TrackTable::Present: return std::to_string(s.frames);
                case TrackTable::Gaps: return s.gaps ? std::to_string(s.gaps) : std::string(kDash);
                case TrackTable::Speed: return s.last > s.first ? format("%.2f", s.umPerFrame) : std::string(kDash);
                case TrackTable::Net: return format("%.2f", s.netUm);
                case TrackTable::Parent: return s.parent ? std::to_string(s.parent) : std::string(kDash);
                case TrackTable::Children: {
                    if (s.children.empty()) return kDash;
                    std::vector<std::string> ids;
                    ids.reserve(s.children.size());
                    for (std::uint32_t c : s.children) ids.push_back(std::to_string(c));
                    return join(ids, ", ");
                }
                default: return {};
            }
        }

        // Sort key: numbers as numbers, not as the text in the cell.
        double sortKey(const TrackSummary& s, int column) {
            switch (column) {
                case TrackTable::Id: return static_cast<double>(s.id);
                case TrackTable::Frames: return static_cast<double>(s.first);
                case TrackTable::Present: return static_cast<double>(s.frames);
                case TrackTable::Gaps: return static_cast<double>(s.gaps);
                case TrackTable::Speed: return s.umPerFrame;
                case TrackTable::Net: return s.netUm;
                case TrackTable::Parent: return static_cast<double>(s.parent);
                case TrackTable::Children: return static_cast<double>(s.children.size());
                default: return 0.0;
            }
        }

    } // namespace

    void TrackTable::setTracks(std::vector<TrackSummary> tracks) {
        tracks_ = std::move(tracks);
        sortDirty_ = true;
        caption_ = caption();
    }

    int TrackTable::rowOf(std::uint32_t id) const {
        for (std::size_t i = 0; i < tracks_.size(); ++i)
            if (tracks_[i].id == id) return static_cast<int>(i);
        return -1;
    }

    std::string TrackTable::caption() const {
        if (tracks_.empty()) return "No tracks: run a tracking step to fill this table.";
        Index gapped = 0;
        for (const TrackSummary& s : tracks_)
            if (s.gaps > 0) ++gapped;
        std::string text = format("%zu tracks · %lld with gaps", tracks_.size(), static_cast<long long>(gapped));
        const Index divisions = countDivisions(tracks_);
        if (divisions > 0) text += format(" · %lld divisions (tracker's estimate, not verified)", static_cast<long long>(divisions));
        return text;
    }

    void TrackTable::resort() {
        order_.resize(tracks_.size());
        for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
        const int column = sortColumn_;
        const bool descending = sortDescending_;
        // stable: rows with equal keys keep the ascending id order they came in
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) {
            const double ka = sortKey(tracks_[static_cast<std::size_t>(a)], column);
            const double kb = sortKey(tracks_[static_cast<std::size_t>(b)], column);
            return descending ? ka > kb : ka < kb;
        });
        sortDirty_ = false;
    }

    TrackTable::Events TrackTable::draw(std::uint32_t selected, bool follow) {
        Events ev;
        if (caption_.empty()) caption_ = caption();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 2.0f || avail.y < 2.0f) return ev;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kBg);

        // --- the caption line: counts on the left, the follow toggle on the right
        const char* followLabel = "Follow selected track";
        const float box = theme::snap(px(14));
        const ImVec2 fs = theme::textSize(followLabel, 12);
        const float checkW = box + px(8) + fs.x;
        const float headH = std::max(box, fs.y) + px(4);
        const float left = origin.x + px(14), right = origin.x + avail.x - px(14);
        const float top = origin.y + px(6);
        const float captionRoom = std::max(px(20), right - left - checkW - px(10));
        const std::string shown = widgets::elideText(caption_, captionRoom, theme::kSmallPx);
        widgets::drawTextIn(dl, ImVec2(left, top), ImVec2(left + captionRoom, top + headH), shown, theme::kSmallPx, theme::kText,
                            theme::Weight::Regular, 0.0f, 0.5f);
        if (shown != caption_) {
            ImGui::SetCursorScreenPos(ImVec2(left, top));
            ImGui::Dummy(ImVec2(captionRoom, headH));
            widgets::tooltip(caption_);
        }
        ImGui::SetCursorScreenPos(ImVec2(std::max(left, right - checkW), top));
        if (widgets::tokenCheck(followLabel, follow)) ev.follow = !follow;
        widgets::tooltip("Keep the selected track centred while the time point changes");

        // --- the table
        const float tableTop = top + headH + px(4);
        const cells::Rect r{ImVec2(origin.x, tableTop), ImVec2(origin.x + avail.x, origin.y + avail.y)};
        if (r.height() < 4.0f) {
            ImGui::Dummy(ImVec2(0, 0));
            return ev;
        }
        if (sortDirty_ || order_.size() != tracks_.size()) resort();
        ImGui::SetCursorScreenPos(r.min);
        cells::pushTableStyle();
        const float headerH = cells::tableHeaderHeight();
        const float rowH = cells::tableRowHeight();
        const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_PadOuterX |
                                      ImGuiTableFlags_Sortable | ImGuiTableFlags_SizingFixedFit |
                                      ImGuiTableFlags_NoSavedSettings;
        bool drawn = false;
        if (ImGui::BeginTable("##tracks", ColumnCount, flags, ImVec2(r.width(), r.height()))) {
            drawn = true;
            ImGui::TableSetupScrollFreeze(0, 1);
            // fixed widths (they include the 6 px cell padding)
            auto width = [&](const char* sample, float extra) {
                return std::max(px(10), theme::textSize(sample, theme::kSmallPx).x + px(extra) - 2.0f * px(6));
            };
            const float widths[ColumnCount] = {width("00000", 46), width("0000 – 0000", 24), width("PRESENT", 24), width("GAPS", 28),
                                               width("µm / FRAME", 24), width("000.00", 30), width("PARENT", 24), width("CHILDREN", 24)};
            for (int c = 0; c < ColumnCount; ++c) {
                // fixed throughout: with a horizontal scroll a stretching column has no width of its own
                ImGuiTableColumnFlags cf = ImGuiTableColumnFlags_WidthFixed;
                if (c == Id) cf |= ImGuiTableColumnFlags_DefaultSort;
                ImGui::TableSetupColumn(kNames[c], cf | ImGuiTableColumnFlags_PreferSortAscending, widths[c]);
            }
            cells::tableHeaders(kTips);
            if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
                if (specs->SpecsDirty) {
                    if (specs->SpecsCount > 0) {
                        sortColumn_ = specs->Specs[0].ColumnIndex;
                        sortDescending_ = specs->Specs[0].SortDirection == ImGuiSortDirection_Descending;
                    }
                    resort();
                    specs->SpecsDirty = false;
                }
            }
            // the selection moved elsewhere (the viewer, the assistant): bring its row into view
            if (selected != shownSelection_) {
                shownSelection_ = selected;
                const int source = selected ? rowOf(selected) : -1;
                if (source >= 0) {
                    const auto at = std::find(order_.begin(), order_.end(), source);
                    const float row = static_cast<float>(at - order_.begin());
                    const float scroll = ImGui::GetScrollY(), visible = ImGui::GetWindowHeight();
                    const float rowTop = headerH + row * rowH;
                    if (rowTop - scroll < headerH) ImGui::SetScrollY(rowTop - headerH);
                    else if (rowTop + rowH - scroll > visible) ImGui::SetScrollY(rowTop + rowH - visible);
                }
            }
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(order_.size()), rowH);
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    const TrackSummary& s = tracks_[static_cast<std::size_t>(order_[static_cast<std::size_t>(row)])];
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                    ImGui::PushID(static_cast<int>(s.id));
                    for (int c = 0; c < ColumnCount; ++c) {
                        if (!ImGui::TableSetColumnIndex(c)) continue;
                        if (c == Id) {
                            // the whole row answers the click; a click on the row
                            // already selected still means "take me there"
                            if (ImGui::Selectable("##row", s.id == selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                                ev.chosen = s.id;
                            ImGui::SameLine(0.0f, 0.0f);
                            widgets::colorChip(theme::fromFloat(labelColor(s.id)), 10, 10);
                            ImGui::SameLine(0.0f, px(6));
                        }
                        // a gap is the first thing to look at: where identity may have been lost
                        const bool accent = c == Gaps && s.gaps > 0;
                        widgets::text(cellText(s, c), theme::kSmallPx, accent ? theme::kAccentText : theme::kText);
                    }
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
        cells::popTableStyle();
        if (drawn) cells::tableHeaderRule(r.min, r.width(), headerH);
        return ev;
    }

} // namespace sirius::app::gui
