// Register: masked translation registration (sirius/registration.hpp) of one
// channel onto another, or of every time point onto a reference, applied as
// an integer-voxel shift.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>
#include <sirius/registration.hpp>

namespace sirius::app {

    namespace {

        constexpr const char* kChannels = "Align channels";
        constexpr const char* kTime = "Align time points to reference";

        Buffer<std::uint8_t> maskOf(const Buffer<float>& v, bool enabled, float level) {
            if (!enabled) return {};
            Buffer<std::uint8_t> m(v.shape());
            for (Index i = 0; i < v.size(); ++i) m.data()[i] = v.data()[i] > level ? 1 : 0;
            return m;
        }

        class RegisterOperation final : public Operation {
        public:
            RegisterOperation() {
                info_.kind = "register";
                info_.name = "Register";
                info_.group = "Combine";
                info_.kindLabel = "COMBINE";
                info_.diagnostics = DiagnosticsKind::Alignment;
                info_.defaultCache = CachePolicy::Memory;
                info_.helpPage = "register";
                info_.params = {
                    choiceParam("mode", "Mode", {kChannels, kTime}, kChannels),
                    channelParam("fixed_channel", "Fixed channel", 0),
                    channelParam("moving_channel", "Moving channel", 1).visibleWhen("mode", {kChannels}),
                    intParam("reference_t", "Reference time point", 0).range(0, 1000000).visibleWhen("mode", {kTime}),
                    doubleListParam("max_shift", "Max shift", {4.0, 32.0, 32.0}).withUnit("voxels").withHelp("Search bound per axis (z, y, x)"),
                    boolParam("mask_background", "Mask background", false),
                    doubleParam("background_level", "Background level", 0.0).range(-1e9, 1e9, 1.0, 2),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                if (p.getString("mode", kChannels) == kTime)
                    return joinSummary({"time points → t " + std::to_string(p.getInt("reference_t")),
                                        "on " + channelName(in, p.getInt("fixed_channel"))});
                return joinSummary({channelName(in, p.getInt("moving_channel", 1)) + " → " + channelName(in, p.getInt("fixed_channel")),
                                    "translation"});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string mode = p.getString("mode", kChannels);
                if (mode == kChannels) {
                    if (in.dims.c < 2) v.errors.push_back("Aligning channels needs at least two channels.");
                    if (p.getInt("fixed_channel") == p.getInt("moving_channel", 1))
                        v.errors.push_back("Fixed and moving channel are the same.");
                } else {
                    if (in.dims.t < 2) v.errors.push_back("Aligning time points needs a time series.");
                    if (p.getInt("reference_t") >= in.dims.t) v.errors.push_back("The reference time point does not exist.");
                }
                if (in.rgb) v.errors.push_back("Register needs intensity channels, not an RGB merge.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const bool alignTime = p.getString("mode", kChannels) == kTime;
                const Index fixedC = p.getInt("fixed_channel"), movingC = p.getInt("moving_channel", 1);
                const Index refT = p.getInt("reference_t");
                const bool useMask = p.getBool("mask_background", false);
                const float level = static_cast<float>(p.getDouble("background_level", 0.0));
                MaskedNccOptions opts;
                const std::vector<double> ms = p.getDoubleList("max_shift");
                if (ms.size() == 3)
                    opts.maxShift = {static_cast<Index>(ms[0]), static_cast<Index>(ms[1]), static_cast<Index>(ms[2])};

                StepOutput out;
                out.meta = outputMeta(p, meta);
                ArrayPtr in = input.materialize([&](double f, const std::string& m) { ctx.report(0.2 * f, m); });
                auto result = std::make_shared<Array5>(in->clone());

                std::vector<TranslationResult> results;
                std::vector<std::string> pairNames;
                Buffer<float> refVol, firstMoving, firstAligned;
                auto shiftInto = [&](const float* src, float* dst, const std::array<Index, 3>& shift) {
                    // voxel q of the aligned image = moving[q - shift]
                    cropPad(src, d.z, d.y, d.x, -shift[0], -shift[1], -shift[2], dst, d.z, d.y, d.x, 0.0f);
                };
                // Labels are one volume per time point, so a time alignment
                // moves them with the images; a channel alignment leaves them
                // where they are (they belong to whichever channel was
                // segmented, usually the fixed one).
                std::shared_ptr<LabelVolume> labels = input.labels ? input.labels->clone() : nullptr;
                auto shiftLabels = [&](Index t, const std::array<Index, 3>& shift) {
                    if (!labels || t >= labels->t()) return;
                    const Index lz = labels->z(), ly = labels->y(), lx = labels->x();
                    std::vector<std::uint32_t> src(labels->volume(t), labels->volume(t) + lz * ly * lx);
                    std::uint32_t* dst = labels->volume(t);
                    for (Index z = 0; z < lz; ++z)
                        for (Index y = 0; y < ly; ++y)
                            for (Index x = 0; x < lx; ++x) {
                                // voxel q of the aligned image = moving[q - shift]
                                const Index sz = z - shift[0], sy = y - shift[1], sx = x - shift[2];
                                dst[(z * ly + y) * lx + x] = (sz >= 0 && sz < lz && sy >= 0 && sy < ly && sx >= 0 && sx < lx)
                                                                 ? src[static_cast<std::size_t>((sz * ly + sy) * lx + sx)]
                                                                 : 0u;
                            }
                };
                Index jobs = alignTime ? d.t : d.t;
                Index done = 0;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(0.2 + 0.75 * static_cast<double>(done++) / std::max<Index>(jobs, 1), "t " + std::to_string(t));
                    if (alignTime && t == refT) continue;
                    Buffer<float> fixed = alignTime ? input.readVolume(fixedC, refT) : input.readVolume(fixedC, t);
                    Buffer<float> moving = input.readVolume(alignTime ? fixedC : movingC, t);
                    Buffer<std::uint8_t> fm = maskOf(fixed, useMask, level), mm = maskOf(moving, useMask, level);
                    TranslationResult r = registerTranslationMasked<float>(fixed.view(), moving.view(), fm.view(), mm.view(), opts);
                    ctx.throwIfCancelled();
                    results.push_back(r);
                    pairNames.push_back(alignTime ? "t " + std::to_string(t) : "t " + std::to_string(t) + " · c " + std::to_string(movingC));
                    if (!r.valid) continue;
                    if (alignTime) {
                        for (Index c = 0; c < d.c; ++c) {
                            ctx.throwIfCancelled();
                            Buffer<float> vol = input.readVolume(c, t);
                            shiftInto(vol.data(), result->volume(c, t).data(), r.integerShift);
                        }
                        shiftLabels(t, r.integerShift);
                    } else {
                        shiftInto(moving.data(), result->volume(movingC, t).data(), r.integerShift);
                    }
                    if (refVol.empty()) {
                        refVol = std::move(fixed);
                        firstMoving = std::move(moving);
                        firstAligned = Buffer<float>(refVol.shape());
                        copy(result->volume(alignTime ? fixedC : movingC, t), firstAligned);
                    }
                }
                out.array = result;
                if (labels && alignTime && labels->statsT() >= 0) labels->recomputeStats(labels->statsT());   // the boxes moved
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.diagnostics = diagnostics(p, meta, results, pairNames, refVol, firstAligned);
                int valid = 0;
                for (const auto& r : results) valid += r.valid ? 1 : 0;
                out.note = std::to_string(valid) + " / " + std::to_string(results.size()) + " pairs registered · CPU";
                ctx.report(1.0, "");
                return out;
            }

        private:
            Diagnostics diagnostics(const ParamSet& p, const DatasetMeta& meta, const std::vector<TranslationResult>& results,
                                    const std::vector<std::string>& names, const Buffer<float>& fixed,
                                    const Buffer<float>& aligned) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Alignment;
                d.summary = summary(p, meta);
                double sum = 0.0, mx = 0.0, ncc = 0.0;
                int n = 0;
                for (const TranslationResult& r : results) {
                    if (!r.valid) continue;
                    const double mag = std::sqrt(r.shift[0] * r.shift[0] + r.shift[1] * r.shift[1] + r.shift[2] * r.shift[2]);
                    sum += mag;
                    mx = std::max(mx, mag);
                    ncc += r.correlation;
                    ++n;
                }
                AlignmentInfo a;
                a.gridRows = 1;
                a.gridCols = static_cast<Index>(std::min<std::size_t>(results.size(), 9));
                for (Index i = 0; i < a.gridCols; ++i) a.tileNames.push_back(names[static_cast<std::size_t>(i)]);
                a.highlightedTile = 0;
                a.shiftStats.push_back({"Pairs", std::to_string(n) + " / " + std::to_string(results.size())});
                a.shiftStats.push_back({"Mean |Δ|", n ? formatNumber(sum / n, 1) + " px" : "—"});
                a.shiftStats.push_back({"Max |Δ|", n ? formatNumber(mx, 1) + " px" : "—"});
                a.shiftStats.push_back({"NCC", n ? formatNumber(ncc / n, 2) : "—"});
                d.facts = a.shiftStats;
                d.alignment = std::move(a);
                DiagnosticTable table;
                table.caption = "Shifts";
                table.header = {"Pair", "Δz", "Δy", "Δx", "NCC"};
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const TranslationResult& r = results[i];
                    table.rows.push_back({names[i], formatNumber(r.shift[0], 2), formatNumber(r.shift[1], 2),
                                          formatNumber(r.shift[2], 2), r.valid ? formatNumber(r.correlation, 2) : "—"});
                    if (!r.valid) table.accentCells.emplace_back(static_cast<int>(i), 4);
                }
                d.table = std::move(table);
                if (!fixed.empty() && !aligned.empty()) {
                    // checkerboard of the middle planes: fixed in the even blocks, aligned moving in the odd ones
                    const Index z = fixed.dim(0) / 2, rows = fixed.dim(1), cols = fixed.dim(2);
                    const float* f = fixed.data() + z * rows * cols;
                    const float* m = aligned.data() + z * rows * cols;
                    auto norm = [](const float* p, Index n) {
                        float mn = std::numeric_limits<float>::infinity(), mxv = -mn;
                        for (Index i = 0; i < n; ++i) {
                            mn = std::min(mn, p[i]);
                            mxv = std::max(mxv, p[i]);
                        }
                        return std::pair<float, float>(mn, mxv > mn ? mxv - mn : 1.0f);
                    };
                    const auto nf = norm(f, rows * cols), nm = norm(m, rows * cols);
                    std::vector<float> board(static_cast<std::size_t>(rows * cols));
                    const Index bw = std::max<Index>(1, cols / 8), bh = std::max<Index>(1, rows / 4);
                    for (Index y = 0; y < rows; ++y)
                        for (Index x = 0; x < cols; ++x) {
                            const bool even = ((x / bw) + (y / bh)) % 2 == 0;
                            const Index i = y * cols + x;
                            board[static_cast<std::size_t>(i)] = even ? (f[i] - nf.first) / nf.second : (m[i] - nm.first) / nm.second;
                        }
                    DiagnosticTab tab{"Alignment", {}};
                    tab.images.push_back(d.addImage(thumbnail(board.data(), rows, cols, 512, "Checkerboard · fixed ⇄ moving", "")));
                    d.tabs.push_back(std::move(tab));
                }
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeRegisterOperation() { return std::make_unique<RegisterOperation>(); }

} // namespace sirius::app
