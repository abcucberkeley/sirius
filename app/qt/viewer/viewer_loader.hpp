#ifndef SIRIUS_APP_VIEWER_LOADER_HPP
#define SIRIUS_APP_VIEWER_LOADER_HPP

// Everything the viewer needs that is too expensive for the GUI thread,
// done on one worker thread of its own.
//
// Two jobs:
//   * prepare(out, c, t)  reads (or, for an in-memory output, walks) one
//     (c, t) volume and returns it with its z maximum projection and its
//     exact value range. The ortho re-slices, the MIP corner and the 3D
//     view all wait on this instead of stalling the window on a multi-
//     gigabyte read through ArraySource.
//   * reduce(...)  turns the volumes of the visible channels into the
//     <= 256^3 8-bit bricks the ray caster uploads, so the first 3D frame
//     is a texture upload and not a reduction of the whole volume inside
//     paintGL.
//
// Lifetime: requests are queued onto a worker QObject that lives on the
// loader's own QThread; every job re-checks the generation it was queued
// with and returns immediately when the viewer has moved on (a new output,
// a new time point, a destroyed loader). Results come back as queued calls
// guarded by a QPointer, and the destructor bumps the generation, quits the
// thread and waits for it, so no job can outlive the loader and ~QObject
// discards any result already posted to it.

#include <array>
#include <atomic>
#include <memory>
#include <set>
#include <vector>

#include <QObject>
#include <QString>
#include <QThread>

#include <sirius/buffer.hpp>

#include "core/operation.hpp"

namespace sirius::app {

    // One channel's volume reduced to a 8-bit brick for the 3D textures.
    struct ReducedVolume {
        std::vector<unsigned char> texels;    // tx * ty * tz, one byte per texel
        int tx = 0, ty = 0, tz = 0;
        std::array<float, 3> color{1.f, 1.f, 1.f};
    };

    class ViewerLoader : public QObject {
        Q_OBJECT
    public:
        explicit ViewerLoader(QObject* parent = nullptr);
        ~ViewerLoader() override;

        // --- volumes ---------------------------------------------------------
        struct Volume {
            std::shared_ptr<const StepOutput> out;
            Index c = 0, t = 0;
            // The read volume; null for an in-memory output, whose volume the
            // display model already has (only `mip` and the range are new).
            std::shared_ptr<Buffer<float>> volume;
            std::shared_ptr<Buffer<float>> mip;
            float lo = 0.0f, hi = 1.0f;      // exact range, NaNs skipped
            bool ok = false;
            QString error;
            qint64 micros = 0;
        };
        // Queues a read of (c, t); a second request for the same (output, c, t)
        // while one is pending does nothing. False = already pending.
        bool prepare(const std::shared_ptr<const StepOutput>& out, Index c, Index t);
        bool pending(const std::shared_ptr<const StepOutput>& out, Index c, Index t) const;
        bool busy() const noexcept { return !pending_.empty() || reductionPending_; }

        // --- 3D textures ------------------------------------------------------
        struct Channel {
            std::shared_ptr<const StepOutput> out;      // keeps an in-memory array alive
            std::shared_ptr<const Buffer<float>> hold;  // keeps a read volume alive
            const float* data = nullptr;                // (z, y, x)
            Index z = 0, y = 0, x = 0;
            float lo = 0.0f, hi = 1.0f;
            std::array<float, 3> color{1.f, 1.f, 1.f};
        };
        struct Reduction {
            quint64 key = 0;
            std::vector<ReducedVolume> channels;
            qint64 micros = 0;
        };
        // Queues the reduction of `channels` into 3D bricks; a newer request
        // replaces an older one that has not started.
        void reduce(quint64 key, std::vector<Channel> channels);
        quint64 reductionKey() const noexcept { return reductionKey_; }

        // Forgets every queued and running job: results that still arrive are
        // dropped. Called when the displayed output changes.
        void cancelAll();

    signals:
        void volumeReady(const sirius::app::ViewerLoader::Volume& v);
        void reductionReady(const sirius::app::ViewerLoader::Reduction& r);
        void volumeProgress(double fraction, const QString& message);

    private:
        struct Job {
            const StepOutput* out = nullptr;
            Index c = 0, t = 0;
            bool operator<(const Job& o) const noexcept {
                return out != o.out ? out < o.out : (c != o.c ? c < o.c : t < o.t);
            }
        };

        QThread thread_;
        QObject* worker_ = nullptr;                 // lives on thread_
        std::shared_ptr<std::atomic<quint64>> generation_;   // shared with the jobs
        quint64 gen_ = 1;
        std::set<Job> pending_;
        bool reductionPending_ = false;
        quint64 reductionKey_ = 0;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_LOADER_HPP
