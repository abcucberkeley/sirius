#include "core/headless.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/cancel.hpp"
#include "core/image_encode.hpp"

// Rendering a step's output as the viewer draws it -- the same display model,
// windows, channel blend, re-slices, projection and label overlay -- into PNG
// or JPEG bytes a model can look at, and a diagnostics image as the
// diagnostics panel draws it.

namespace sirius::app {

    using json = nlohmann::json;

    namespace {

        // The long edge of an image a client is sent at most (what vision
        // models take without scaling it themselves).
        constexpr int kMaxEdge = 1568;
        // A PNG larger than this is sent as JPEG when no format was asked for.
        constexpr std::size_t kPngFallbackBytes = std::size_t{1} << 20;
        // Pixels between the tiles of a grid and the panels of a channel layout.
        constexpr int kGap = 4;
        constexpr std::array<std::uint8_t, 3> kGapColor{0x40, 0x40, 0x40};

        [[noreturn]] void invalid(const std::string& message, const std::string& hint = {}) {
            throw ToolFailure("invalid_argument", message, hint);
        }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        bool isIntegral(const json& v) {
            return v.is_number_integer() || (v.is_number_float() && v.get<double>() == std::floor(v.get<double>()));
        }

        std::int64_t integer(const json& v, const char* what, std::int64_t min) {
            if (!v.is_number() || !isIntegral(v)) invalid(std::string("'") + what + "' must be an integer");
            const std::int64_t i = v.is_number_integer() ? v.get<std::int64_t>() : static_cast<std::int64_t>(v.get<double>());
            if (i < min) invalid(std::string("'") + what + "' must be at least " + std::to_string(min));
            return i;
        }

        double number(const json& v, const char* what) {
            if (!v.is_number()) invalid(std::string("'") + what + "' must be a number");
            return v.get<double>();
        }

        bool boolean(const json& v, const char* what) {
            if (!v.is_boolean()) invalid(std::string("'") + what + "' must be true or false");
            return v.get<bool>();
        }

        std::string text(const json& v, const char* what) {
            if (!v.is_string()) invalid(std::string("'") + what + "' must be a string");
            return lower(v.get<std::string>());
        }

        const json* find(const json& a, const char* key) {
            const auto it = a.find(key);
            return it == a.end() || it->is_null() ? nullptr : &*it;
        }

        int ceilDiv(Index a, int b) { return static_cast<int>((a + b - 1) / b); }

        // An RGB picture, rows top to bottom.
        struct Rgb {
            int width = 0, height = 0;
            std::vector<std::uint8_t> bytes;
            void resize(int w, int h, std::array<std::uint8_t, 3> fill) {
                width = w;
                height = h;
                bytes.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u, 0);
                for (std::size_t i = 0; i < bytes.size(); i += 3) {
                    bytes[i] = fill[0];
                    bytes[i + 1] = fill[1];
                    bytes[i + 2] = fill[2];
                }
            }
            std::uint8_t* at(int x, int y) { return bytes.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3u; }
        };

        // An Image pixel is 0xAABBGGRR; the bytes R, G, B go into the RGB picture.
        void blit(const display::Image& img, Rgb& dst, int ox, int oy) {
            for (int y = 0; y < img.height && oy + y < dst.height; ++y) {
                const std::uint32_t* src = img.scanLine(y);
                for (int x = 0; x < img.width && ox + x < dst.width; ++x) {
                    std::uint8_t* p = dst.at(ox + x, oy + y);
                    p[0] = static_cast<std::uint8_t>(src[x] & 0xffu);
                    p[1] = static_cast<std::uint8_t>((src[x] >> 8) & 0xffu);
                    p[2] = static_cast<std::uint8_t>((src[x] >> 16) & 0xffu);
                }
            }
        }

        // Nearest-neighbour stretch of one axis by `scale`: the rows of an XZ
        // re-slice (z) or the columns of a YZ one, so a voxel is as tall as it
        // is physically, as the viewer's panes draw it.
        display::Image stretched(const display::Image& img, double scale, bool rows) {
            if (std::abs(scale - 1.0) < 1e-6 || img.isNull()) return img;
            display::Image out;
            const int w = rows ? img.width : std::max(1, static_cast<int>(std::lround(img.width * scale)));
            const int h = rows ? std::max(1, static_cast<int>(std::lround(img.height * scale))) : img.height;
            out.resize(w, h);
            for (int y = 0; y < h; ++y) {
                const int sy = rows ? std::min(img.height - 1, static_cast<int>((y + 0.5) / scale)) : y;
                const std::uint32_t* src = img.scanLine(sy);
                std::uint32_t* dst = out.scanLine(y);
                for (int x = 0; x < w; ++x) dst[x] = src[rows ? x : std::min(img.width - 1, static_cast<int>((x + 0.5) / scale))];
            }
            return out;
        }

        // The part [x0, x0 + w) x [y0, y0 + h) of an image, clamped.
        display::Image cropped(const display::Image& img, int x0, int y0, int w, int h) {
            x0 = std::clamp(x0, 0, std::max(img.width - 1, 0));
            y0 = std::clamp(y0, 0, std::max(img.height - 1, 0));
            w = std::clamp(w, 1, img.width - x0);
            h = std::clamp(h, 1, img.height - y0);
            display::Image out;
            out.resize(w, h);
            for (int y = 0; y < h; ++y) std::copy_n(img.scanLine(y0 + y) + x0, w, out.scanLine(y));
            return out;
        }

        std::vector<std::uint8_t> encoded(const Rgb& img, const std::string& format) {
            return format == "jpeg" ? encodeJpeg(img.bytes.data(), img.width, img.height, 3, 90)
                                    : encodePng(img.bytes.data(), img.width, img.height, 3);
        }

        // A display window in a caption: [lo, hi, gamma].
        json windowJson(const display::DisplayWindow& w) { return json::array({w.lo, w.hi, w.gamma}); }

    } // namespace

    // --- the request -------------------------------------------------------------------

    RenderRequest renderRequestFromJson(const json& a) {
        RenderRequest r;
        if (!a.is_object()) return r;
        if (const json* v = find(a, "step"); v && v->is_number()) r.step = static_cast<int>(integer(*v, "step", 1)) - 1;
        if (const json* v = find(a, "plane")) {
            r.plane = text(*v, "plane");
            if (r.plane != "xy" && r.plane != "xz" && r.plane != "yz" && r.plane != "mip")
                invalid("'plane' must be xy, xz, yz or mip");
        }
        if (const json* v = find(a, "z")) {
            if (v->is_array()) {
                if (v->empty() || v->size() > 16) invalid("'z' takes one plane or a list of 1 to 16 planes");
                for (const json& z : *v) r.z.push_back(static_cast<Index>(integer(z, "z", 0)));
            } else {
                r.z.push_back(static_cast<Index>(integer(*v, "z", 0)));
            }
            if (r.z.size() > 1 && r.plane != "xy") invalid("a grid of several z planes is drawn for plane xy only");
        }
        if (const json* v = find(a, "t")) r.t = static_cast<Index>(integer(*v, "t", 0));
        if (const json* v = find(a, "y")) r.y = static_cast<Index>(integer(*v, "y", 0));
        if (const json* v = find(a, "x")) r.x = static_cast<Index>(integer(*v, "x", 0));
        if (const json* v = find(a, "channels")) {
            if (!v->is_array()) invalid("'channels' must be a list of channel indices");
            for (const json& c : *v) r.channels.push_back(static_cast<Index>(integer(c, "channels", 0)));
        }
        if (const json* v = find(a, "layout")) {
            r.layout = text(*v, "layout");
            if (r.layout != "blend" && r.layout != "channels") invalid("'layout' must be blend or channels");
        }
        if (const json* v = find(a, "window")) {
            r.window = text(*v, "window");
            if (r.window != "auto" && r.window != "full") invalid("'window' must be auto or full", "per-channel windows go in 'windows'");
        }
        if (const json* v = find(a, "windows")) {
            if (!v->is_array()) invalid("'windows' must be a list of {channel, lo, hi, gamma}");
            for (const json& w : *v) {
                if (!w.is_object() || !w.contains("channel") || !w.contains("lo") || !w.contains("hi"))
                    invalid("each entry of 'windows' needs channel, lo and hi");
                RenderRequest::ChannelWindow cw;
                cw.channel = static_cast<Index>(integer(w["channel"], "windows.channel", 0));
                cw.lo = static_cast<float>(number(w["lo"], "windows.lo"));
                cw.hi = static_cast<float>(number(w["hi"], "windows.hi"));
                if (w.contains("gamma") && !w["gamma"].is_null()) cw.gamma = static_cast<float>(number(w["gamma"], "windows.gamma"));
                if (!(cw.hi > cw.lo)) invalid("a window needs hi > lo");
                if (!(cw.gamma > 0.0f)) invalid("a window's gamma must be positive");
                r.windows.push_back(cw);
            }
        }
        if (const json* v = find(a, "labels")) r.labels = boolean(*v, "labels");
        if (const json* v = find(a, "label_opacity")) {
            r.labelOpacity = number(*v, "label_opacity");
            if (r.labelOpacity < 0.0 || r.labelOpacity > 1.0) invalid("'label_opacity' must be within 0..1");
        }
        if (const json* v = find(a, "label")) r.label = static_cast<std::uint32_t>(integer(*v, "label", 0));
        if (const json* v = find(a, "solo")) r.solo = boolean(*v, "solo");
        if (const json* v = find(a, "region")) {
            if (!v->is_array() || v->size() != 4) invalid("'region' must be [x, y, w, h] in voxels of the plane");
            for (std::size_t i = 0; i < 4; ++i) r.region[i] = static_cast<int>(integer((*v)[i], "region", i < 2 ? 0 : 1));
        }
        if (const json* v = find(a, "max_size")) r.maxSize = static_cast<int>(std::min<std::int64_t>(integer(*v, "max_size", 0), kMaxEdge));
        if (const json* v = find(a, "physical_z")) r.physicalZ = boolean(*v, "physical_z");
        if (const json* v = find(a, "format")) {
            r.format = text(*v, "format");
            if (r.format == "jpg") r.format = "jpeg";
            if (r.format != "png" && r.format != "jpeg" && !r.format.empty()) invalid("'format' must be png or jpeg");
        }
        return r;
    }

    // --- a step's output ---------------------------------------------------------------

    RenderResult renderOutput(std::shared_ptr<const StepOutput> out, int stepIndex, const RenderRequest& r,
                              display::DisplayModel& model, const std::function<bool()>& cancelled) {
        const std::string stepName = "step " + std::to_string(stepIndex + 1);
        if (!out || !(out->array || out->source))
            throw ToolFailure("not_computed", stepName + " has no output to render", "run it first, or pass run:true");
        const DatasetMeta& meta = out->meta;
        const Dims5 d = meta.dims;
        if (d.c <= 0 || d.t <= 0 || d.z <= 0 || d.y <= 0 || d.x <= 0) throw ToolFailure("not_computed", stepName + "'s output is empty");
        const std::string& plane = r.plane;
        if (plane != "xy" && plane != "xz" && plane != "yz" && plane != "mip") invalid("'plane' must be xy, xz, yz or mip");
        if (r.t < 0 || r.t >= d.t) invalid("t " + std::to_string(r.t) + " is outside 0.." + std::to_string(d.t - 1));
        std::vector<Index> zs = r.z.empty() ? std::vector<Index>{d.z / 2} : r.z;
        if (zs.size() > 1 && plane != "xy") invalid("a grid of several z planes is drawn for plane xy only");
        for (Index z : zs)
            if (z < 0 || z >= d.z) invalid("z " + std::to_string(z) + " is outside 0.." + std::to_string(d.z - 1));
        const Index y = r.y.value_or(d.y / 2), x = r.x.value_or(d.x / 2);
        if (y < 0 || y >= d.y) invalid("y " + std::to_string(y) + " is outside 0.." + std::to_string(d.y - 1));
        if (x < 0 || x >= d.x) invalid("x " + std::to_string(x) + " is outside 0.." + std::to_string(d.x - 1));
        std::vector<Index> visible;
        if (r.channels.empty()) {
            for (Index c = 0; c < d.c; ++c) visible.push_back(c);
        } else {
            for (Index c : r.channels) {
                if (c < 0 || c >= d.c) invalid("channel " + std::to_string(c) + " is outside 0.." + std::to_string(d.c - 1));
                if (std::find(visible.begin(), visible.end(), c) == visible.end()) visible.push_back(c);
            }
        }
        for (const RenderRequest::ChannelWindow& w : r.windows)
            if (w.channel < 0 || w.channel >= d.c) invalid("the window's channel " + std::to_string(w.channel) + " does not exist");
        const bool channelsLayout = r.layout == "channels";
        if (channelsLayout && zs.size() > 1) invalid("the channels layout draws one z plane");
        const auto isCancelled = [&cancelled] { return cancelled && cancelled(); };

        // The channels layout draws every channel in grey: the same output
        // with white channel colours, so the model's blend is a grey ramp.
        std::shared_ptr<const StepOutput> shown = out;
        if (channelsLayout) {
            auto grey = std::make_shared<StepOutput>(*out);
            for (ChannelInfo& ch : grey->meta.channels) ch.color = {1.0f, 1.0f, 1.0f};
            grey->meta.rgb = false;
            shown = std::move(grey);
        }
        model.setOutput(shown);
        if (!model.valid()) throw ToolFailure("not_computed", stepName + " has no output to render", "run it first, or pass run:true");
        // Every request starts from the automatic (or full-range) windows:
        // this also drops the explicit windows an earlier request set.
        model.setWindowMode(r.window == "full" ? display::DisplayModel::WindowMode::Full : display::DisplayModel::WindowMode::Auto);
        for (const RenderRequest::ChannelWindow& w : r.windows) model.setWindow(w.channel, display::DisplayWindow{w.lo, w.hi, w.gamma});

        ViewState vs;
        vs.channelVisible.assign(static_cast<std::size_t>(d.c), false);
        for (Index c : visible) vs.channelVisible[static_cast<std::size_t>(c)] = true;
        vs.labelOpacity = r.labelOpacity;
        vs.selectedLabel = r.label;
        vs.soloLabel = r.solo;
        // the viewer draws no labels over the projection
        const bool labels = plane != "mip" && model.hasLabels() && r.labels.value_or(true);
        vs.labels = labels;

        // The plane's own extents in voxels: (columns, rows).
        const Index cols = plane == "yz" ? d.z : d.x;
        const Index rows = plane == "xz" ? d.z : d.y;
        display::RectI region{0, 0, static_cast<int>(cols), static_cast<int>(rows)};
        if (r.region[2] > 0 && r.region[3] > 0) {
            region = display::RectI{r.region[0], r.region[1], r.region[2], r.region[3]}.intersected(region);
            if (region.empty()) invalid("the region lies outside the " + plane + " plane (" + std::to_string(cols) + " x " + std::to_string(rows) + ")");
        }

        // What the renderers read that a plane does not give: the whole (c, t)
        // volume for a re-slice, the projection for the MIP. The full-range
        // window of an xy plane is the volume's exact range too, which the
        // projection's pass installs; without it the model stands in a range
        // sampled from five planes. A channel with an explicit window needs none.
        const auto hasExplicitWindow = [&r](Index c) {
            return std::any_of(r.windows.begin(), r.windows.end(), [c](const RenderRequest::ChannelWindow& w) { return w.channel == c; });
        };
        for (Index c : visible) {
            if (isCancelled()) throw CancelledError();
            if (plane == "xz" || plane == "yz") {
                if (!model.prepareVolumeSync(c, r.t, cancelled))
                    throw ToolFailure("too_large", "a (c, t) volume of " + stepName + " is larger than the 3 GiB a re-slice may read",
                                      "render plane xy or mip instead");
            } else if (plane == "mip" || (r.window == "full" && !hasExplicitWindow(c))) {
                model.prepareProjectionSync(c, r.t, cancelled);
            }
        }

        // A voxel's height over its width in the re-slices, when drawn physically.
        double aspect = 1.0;
        if ((plane == "xz" || plane == "yz") && r.physicalZ && meta.voxelUm[0] > 0.0 && meta.voxelUm[2] > 0.0)
            aspect = meta.voxelUm[2] / meta.voxelUm[0];
        const int tiles = static_cast<int>(zs.size());
        const int gridCols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(tiles))));
        const int gridRows = (tiles + gridCols - 1) / gridCols;
        const int panels = channelsLayout ? static_cast<int>(visible.size()) : 1;
        // One tile's size at a factor, after the physical stretch.
        const auto tileSize = [&](int f) {
            int w = ceilDiv(region.w, f), h = ceilDiv(region.h, f);
            if (plane == "xz") h = std::max(1, static_cast<int>(std::lround(h * aspect)));
            if (plane == "yz") w = std::max(1, static_cast<int>(std::lround(w * aspect)));
            return std::make_pair(w, h);
        };
        const auto compositeSize = [&](int f) {
            const auto [tw, th] = tileSize(f);
            const int cellW = panels * tw + (panels - 1) * kGap;
            return std::make_pair(gridCols * cellW + (gridCols - 1) * kGap, gridRows * th + (gridRows - 1) * kGap);
        };
        const int cap = r.maxSize <= 0 ? kMaxEdge : std::min(r.maxSize, kMaxEdge);
        int factor = 1;
        {
            const auto [w1, h1] = compositeSize(1);
            factor = std::max(1, std::max(w1, h1) / cap);
            while (factor < (1 << 20)) {
                const auto [w, h] = compositeSize(factor);
                if (std::max(w, h) <= cap) break;
                ++factor;
            }
        }

        const auto compose = [&](int f) {
            const auto [tw, th] = tileSize(f);
            const auto [cw, ch] = compositeSize(f);
            Rgb img;
            img.resize(cw, ch, kGapColor);
            display::Image tile;
            for (int i = 0; i < tiles; ++i) {
                const int gx = (i % gridCols) * (panels * tw + (panels - 1) * kGap + kGap);
                const int gy = (i / gridCols) * (th + kGap);
                for (int p = 0; p < panels; ++p) {
                    if (isCancelled()) throw CancelledError();
                    ViewState v = vs;
                    if (channelsLayout) {
                        v.channelVisible.assign(static_cast<std::size_t>(d.c), false);
                        v.channelVisible[static_cast<std::size_t>(visible[static_cast<std::size_t>(p)])] = true;
                    }
                    display::Image drawn;
                    if (plane == "xy") {
                        model.renderXY(r.t, zs[static_cast<std::size_t>(i)], v, f, tile, region);
                        if (labels) model.overlayLabelsXY(r.t, zs[static_cast<std::size_t>(i)], f, v, tile, region);
                        drawn = tile;
                    } else if (plane == "xz") {
                        model.renderXZ(r.t, y, v, f, tile, region);
                        if (labels) model.overlayLabelsXZ(r.t, y, f, v, tile, region);
                        drawn = stretched(tile, aspect, true);
                    } else if (plane == "yz") {
                        model.renderYZ(r.t, x, v, f, tile, region);
                        if (labels) model.overlayLabelsYZ(r.t, x, f, v, tile, region);
                        drawn = stretched(tile, aspect, false);
                    } else {
                        model.renderMIP(r.t, v, f, tile);
                        drawn = cropped(tile, region.x / f, region.y / f, ceilDiv(region.w, f), ceilDiv(region.h, f));
                    }
                    blit(drawn, img, gx + p * (tw + kGap), gy);
                }
            }
            return img;
        };

        RenderResult result;
        Rgb img;
        std::string format;
        for (int attempt = 0;; ++attempt) {
            img = compose(factor);
            format = r.format.empty() ? "png" : r.format;
            result.bytes = encoded(img, format);
            if (r.format.empty() && result.bytes.size() > kPngFallbackBytes) {
                format = "jpeg";
                result.bytes = encoded(img, format);
            }
            if (result.bytes.size() <= r.maxBytes) break;
            if (attempt == 3)
                throw ToolFailure("too_large",
                                  "the image is " + std::to_string(result.bytes.size()) + " bytes, over the budget of " +
                                      std::to_string(r.maxBytes),
                                  "lower max_size or pick a region");
            factor *= 2;
        }

        result.mimeType = format == "jpeg" ? "image/jpeg" : "image/png";
        result.width = img.width;
        result.height = img.height;
        result.factor = factor;
        const double dx = meta.voxelUm[0] * factor, dy = meta.voxelUm[1] * factor, dz = meta.voxelUm[2] * factor;
        json pixelUm = json::array({dx, dy});
        if (plane == "xz") pixelUm = json::array({dx, dz / aspect});
        if (plane == "yz") pixelUm = json::array({dz / aspect, dy});
        json channels = json::array();
        for (Index c : visible) {
            const ChannelInfo ch = static_cast<std::size_t>(c) < meta.channels.size() ? meta.channels[static_cast<std::size_t>(c)] : ChannelInfo{};
            channels.push_back({{"index", c}, {"label", ch.label}, {"color", ch.hexColor()}, {"window", windowJson(model.window(c, r.t))}});
        }
        json z = nullptr;
        if (plane == "xy") z = zs.size() == 1 ? json(zs.front()) : json(zs);
        result.caption = {{"step", stepIndex + 1},
                          {"rendered_step", stepIndex + 1},
                          {"plane", plane},
                          {"t", r.t},
                          {"z", z},
                          {"y", plane == "xz" ? json(y) : json(nullptr)},
                          {"x", plane == "yz" ? json(x) : json(nullptr)},
                          {"width", result.width},
                          {"height", result.height},
                          {"factor", factor},
                          {"pixel_um", pixelUm},
                          {"region", {region.x, region.y, region.w, region.h}},
                          {"layout", channelsLayout ? "channels" : "blend"},
                          {"channels", channels},
                          {"labels_drawn", labels},
                          {"format", format},
                          {"bytes", result.bytes.size()}};
        return result;
    }

    // --- a diagnostics image -------------------------------------------------------------

    RenderResult renderDiagnosticImage(const DiagnosticImage& image, int maxSize, std::size_t maxBytes) {
        const Index rows = image.rows, cols = image.cols;
        if (rows <= 0 || cols <= 0 || image.values.size() < static_cast<std::size_t>(rows * cols))
            invalid("the diagnostics image '" + image.title + "' is empty");
        const Index n = rows * cols;
        // The panel's window (diagnostic_cells.cpp): 0.5 and 99.8 percentiles
        // of a bounded sample, the full range when they meet.
        std::vector<float> sample;
        const Index stride = std::max<Index>(1, n / 65536);
        for (Index i = 0; i < n; i += stride) {
            const float v = image.values[static_cast<std::size_t>(i)];
            if (std::isfinite(v)) sample.push_back(v);
        }
        float lo = 0.0f, hi = 1.0f;
        if (!sample.empty()) {
            const auto rank = [&](double frac) { return static_cast<std::ptrdiff_t>(std::llround(frac * static_cast<double>(sample.size() - 1))); };
            const std::ptrdiff_t kLo = rank(0.005), kHi = rank(0.998);
            std::nth_element(sample.begin(), sample.begin() + kLo, sample.end());
            lo = sample[static_cast<std::size_t>(kLo)];
            std::nth_element(sample.begin() + kLo, sample.begin() + kHi, sample.end());
            hi = sample[static_cast<std::size_t>(kHi)];
            if (!(hi > lo)) {
                const auto [mn, mx] = std::minmax_element(sample.begin(), sample.end());
                lo = *mn;
                hi = *mx;
            }
            if (!(hi > lo)) hi = lo + 1.0f;
        }
        const int cap = maxSize <= 0 ? kMaxEdge : std::min(maxSize, kMaxEdge);
        const float scale = 255.0f / (hi - lo);
        // The picture at a reduction factor, and its marks in its pixels.
        const auto draw = [&](int f, json& marks) {
            const int w = ceilDiv(cols, f), h = ceilDiv(rows, f);
            Rgb img;
            img.resize(w, h, {0, 0, 0});
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    // a box average when reduced, as the panel's texture does
                    double acc = 0.0;
                    int count = 0;
                    for (Index sy = static_cast<Index>(y) * f; sy < std::min<Index>(rows, static_cast<Index>(y + 1) * f); ++sy)
                        for (Index sx = static_cast<Index>(x) * f; sx < std::min<Index>(cols, static_cast<Index>(x + 1) * f); ++sx) {
                            const float v = image.values[static_cast<std::size_t>(sy * cols + sx)];
                            if (!std::isfinite(v)) continue;
                            acc += v;
                            ++count;
                        }
                    const float t = count > 0 ? std::clamp((static_cast<float>(acc / count) - lo) * scale, 0.0f, 255.0f) : 0.0f;
                    std::uint8_t* p = img.at(x, y);
                    p[0] = p[1] = p[2] = static_cast<std::uint8_t>(t + 0.5f);
                }

            // The marks, in the panel's colours: the accent, or the viewer's text.
            const auto blend = [&img](int x, int y, std::array<std::uint8_t, 3> c, float alpha) {
                if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
                std::uint8_t* p = img.at(x, y);
                for (int k = 0; k < 3; ++k) p[k] = static_cast<std::uint8_t>(std::lround(p[k] * (1.0f - alpha) + c[static_cast<std::size_t>(k)] * alpha));
            };
            marks = json::array();
            for (const DiagnosticMark& m : image.marks) {
                const std::array<std::uint8_t, 3> c = m.accent ? std::array<std::uint8_t, 3>{0xec, 0x30, 0x13} : std::array<std::uint8_t, 3>{0xea, 0xe9, 0xe9};
                const double cx = m.x / f, cy = m.y / f;
                const double radius = std::max(2.0, m.radius / f);
                const char* kind = "cross";
                if (m.kind == DiagnosticMark::Kind::Cross) {
                    const int ix = static_cast<int>(std::lround(cx)), iy = static_cast<int>(std::lround(cy));
                    for (int k = -5; k <= 5; ++k) {
                        blend(ix + k, iy, c, 1.0f);
                        blend(ix, iy + k, c, 1.0f);
                    }
                } else {
                    kind = m.kind == DiagnosticMark::Kind::Circle ? "circle" : "ring";
                    const int r = static_cast<int>(std::ceil(radius)) + 1;
                    for (int yy = static_cast<int>(cy) - r; yy <= static_cast<int>(cy) + r; ++yy)
                        for (int xx = static_cast<int>(cx) - r; xx <= static_cast<int>(cx) + r; ++xx) {
                            const double dist = std::hypot(xx + 0.5 - cx, yy + 0.5 - cy);
                            if (std::abs(dist - radius) < 0.6) blend(xx, yy, c, 1.0f);
                            else if (m.kind == DiagnosticMark::Kind::Circle && dist < radius) blend(xx, yy, c, 0.35f);
                        }
                }
                json mj = {{"kind", kind}, {"x", cx}, {"y", cy}};
                if (m.kind != DiagnosticMark::Kind::Cross) mj["radius"] = radius;
                if (!m.text.empty()) mj["text"] = m.text;
                marks.push_back(std::move(mj));
            }
            return img;
        };

        // Without marks the picture is grey, and a one-channel PNG a third of
        // the data. Over the budget (a noisy spectrum compresses badly) it is
        // reduced further, as render does, before it is refused.
        int f = std::max(1, static_cast<int>((std::max(rows, cols) + cap - 1) / cap));
        Rgb img;
        json marks;
        std::vector<std::uint8_t> bytes;
        for (int attempt = 0;; ++attempt) {
            img = draw(f, marks);
            if (image.marks.empty()) {
                std::vector<std::uint8_t> grey(static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height));
                for (std::size_t i = 0; i < grey.size(); ++i) grey[i] = img.bytes[3 * i];
                bytes = encodePng(grey.data(), img.width, img.height, 1);
            } else {
                bytes = encodePng(img.bytes.data(), img.width, img.height, 3);
            }
            if (bytes.size() <= maxBytes) break;
            if (attempt == 3 || (img.width <= 1 && img.height <= 1))
                throw ToolFailure("too_large",
                                  "the image is " + std::to_string(bytes.size()) + " bytes, over the budget of " + std::to_string(maxBytes),
                                  "lower max_size");
            f *= 2;
        }

        RenderResult result;
        result.bytes = std::move(bytes);
        result.mimeType = "image/png";
        result.width = img.width;
        result.height = img.height;
        result.factor = f;
        result.caption = {{"title", image.title},
                          {"meta", image.meta},
                          {"width", img.width},
                          {"height", img.height},
                          {"factor", f},
                          {"log_scale", image.logScale},
                          {"window", {lo, hi}},
                          {"marks", marks},
                          {"format", "png"},
                          {"bytes", result.bytes.size()}};
        return result;
    }

} // namespace sirius::app
