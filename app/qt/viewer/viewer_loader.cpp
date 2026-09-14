#include "qt/viewer/viewer_loader.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>
#include <limits>

#include <QElapsedTimer>
#include <QPointer>

#include "core/array_source.hpp"
#include "qt/viewer/viewer_constants.hpp"

namespace sirius::app {

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
        // to 0..255. This is the loop that used to sit inside paintGL.
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
            out.texels.assign(static_cast<std::size_t>(out.tx) * out.ty * out.tz, 0);
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
                        out.texels[(static_cast<std::size_t>(z) * out.ty + y) * out.tx + x] =
                            static_cast<unsigned char>(v > 255.0f ? 255 : (v > 0.0f ? static_cast<int>(v) : 0));
                    }
            return out;
        }
    } // namespace

    ViewerLoader::ViewerLoader(QObject* parent)
        : QObject(parent), generation_(std::make_shared<std::atomic<quint64>>(1)) {
        worker_ = new QObject;
        worker_->moveToThread(&thread_);
        connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
        thread_.setObjectName(QStringLiteral("sirius-viewer-loader"));
        thread_.start(QThread::LowPriority);
    }

    ViewerLoader::~ViewerLoader() {
        // A running job sees the bump at its next check and returns; anything
        // still queued never starts. wait() then guarantees no job touches
        // this object (or the data it holds) after the destructor.
        generation_->fetch_add(1);
        thread_.quit();
        thread_.wait();
    }

    void ViewerLoader::cancelAll() {
        generation_->fetch_add(1);
        gen_ = generation_->load();
        pending_.clear();
        reductionPending_ = false;
        reductionKey_ = 0;
    }

    bool ViewerLoader::pending(const std::shared_ptr<const StepOutput>& out, Index c, Index t) const {
        return pending_.count(Job{out.get(), c, t}) != 0;
    }

    bool ViewerLoader::prepare(const std::shared_ptr<const StepOutput>& out, Index c, Index t) {
        if (!out) return false;
        const Job job{out.get(), c, t};
        if (!pending_.insert(job).second) return false;   // already queued or running
        const quint64 gen = gen_;
        auto generation = generation_;
        QPointer<ViewerLoader> self(this);
        QMetaObject::invokeMethod(
            worker_,
            [self, generation, gen, out, c, t] {
                if (generation->load() != gen) return;   // the viewer moved on
                QElapsedTimer clock;
                clock.start();
                Volume result;
                result.out = out;
                result.c = c;
                result.t = t;
                const Dims5& d = out->meta.dims;
                try {
                    const float* vol = nullptr;
                    if (out->array) {
                        vol = out->array->plane(c, t, 0);
                    } else if (out->source) {
                        auto buf = std::make_shared<Buffer<float>>(Shape{d.z, d.y, d.x});
                        out->source->readVolume(c, t, buf->data(), [&](double f, const std::string& m) {
                            if (generation->load() != gen || !self) return;
                            emit self->volumeProgress(0.9 * f, QString::fromStdString(m));
                        });
                        vol = buf->data();
                        result.volume = std::move(buf);
                    }
                    if (generation->load() != gen) return;
                    if (vol) {
                        auto mip = std::make_shared<Buffer<float>>(Shape{d.y, d.x});
                        projectAndRange(vol, d.z, d.y, d.x, mip->data(), result.lo, result.hi);
                        result.mip = std::move(mip);
                        result.ok = true;
                    } else {
                        result.error = QStringLiteral("no data source");
                    }
                } catch (const std::exception& e) {
                    result.ok = false;
                    result.error = QString::fromUtf8(e.what());
                }
                result.micros = clock.nsecsElapsed() / 1000;
                if (generation->load() != gen || !self) return;
                QMetaObject::invokeMethod(
                    self.data(),
                    [self, generation, gen, result = std::move(result)] {
                        if (!self || generation->load() != gen) return;
                        self->pending_.erase(Job{result.out.get(), result.c, result.t});
                        emit self->volumeReady(result);
                    },
                    Qt::QueuedConnection);
            },
            Qt::QueuedConnection);
        return true;
    }

    void ViewerLoader::reduce(quint64 key, std::vector<Channel> channels) {
        if (key == reductionKey_ && reductionPending_) return;
        reductionKey_ = key;
        reductionPending_ = true;
        const quint64 gen = gen_;
        auto generation = generation_;
        QPointer<ViewerLoader> self(this);
        QMetaObject::invokeMethod(
            worker_,
            [self, generation, gen, key, channels = std::move(channels)] {
                if (generation->load() != gen) return;
                QElapsedTimer clock;
                clock.start();
                Reduction result;
                result.key = key;
                for (const Channel& ch : channels) {
                    if (generation->load() != gen) return;
                    result.channels.push_back(reduceChannel(ch));
                }
                result.micros = clock.nsecsElapsed() / 1000;
                if (generation->load() != gen || !self) return;
                QMetaObject::invokeMethod(
                    self.data(),
                    [self, generation, gen, result = std::move(result)] {
                        if (!self || generation->load() != gen) return;
                        if (self->reductionKey_ == result.key) self->reductionPending_ = false;
                        emit self->reductionReady(result);
                    },
                    Qt::QueuedConnection);
            },
            Qt::QueuedConnection);
    }

} // namespace sirius::app
