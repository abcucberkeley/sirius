#ifndef SIRIUS_IMGUI_VIEWER_LOADER_HPP
#define SIRIUS_IMGUI_VIEWER_LOADER_HPP

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
//     the frame.
//
// Lifetime: requests are queued for the loader's own std::thread, first in
// first out. Every job re-checks the generation it was queued with, which
// cancelAll() (a new output) and the destructor move, and returns as soon as
// it changed. A volume read for a time point the viewer no longer shows is
// dropped by retain(): taken off the queue before it starts, or, when it is
// already running, left to finish a read that cannot be interrupted with no
// progress posted. What such a read produced still comes back (the viewer
// caches its projection), but it no longer answers a request. Results come
// back through the poster the loader was given (Bridge::post: the GUI
// thread, at the start of a frame) guarded by a shared "alive" flag, and
// the destructor bumps the generation, clears the flag and joins the
// thread, so no job can outlive the loader and a result already posted is
// dropped when it arrives.

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sirius/buffer.hpp>

#include "core/operation.hpp"

namespace sirius::app::gui {

    // One channel's volume reduced to a 8-bit brick for the 3D textures.
    struct ReducedVolume {
        std::vector<unsigned char> texels;    // tx * ty * tz, one byte per texel
        int tx = 0, ty = 0, tz = 0;
        std::array<float, 3> color{1.f, 1.f, 1.f};
    };

    class ViewerLoader {
    public:
        // Runs a function on the GUI thread; callable from any thread.
        using Poster = std::function<void(std::function<void()>)>;

        explicit ViewerLoader(Poster post);
        ~ViewerLoader();
        ViewerLoader(const ViewerLoader&) = delete;
        ViewerLoader& operator=(const ViewerLoader&) = delete;

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
            std::string error;
            long long micros = 0;
        };
        // Queues a read of (c, t); a second request for the same (output, c, t)
        // while one is pending does nothing. False = already pending.
        bool prepare(const std::shared_ptr<const StepOutput>& out, Index c, Index t);
        bool pending(const std::shared_ptr<const StepOutput>& out, Index c, Index t) const;
        // Drops every pending read of `out` whose t is not in `keepT`: the
        // queued ones never start, the running one posts no progress (only
        // what it read, once done, not as the request's answer), and each
        // can be asked for again at once. Reads the viewer passed over
        // (scrubbing, play) would otherwise hold up the frame on screen.
        void retain(const StepOutput* out, const std::vector<Index>& keepT);
        // Volume reads queued or running. A reduction is not counted: the
        // viewer's loading state is about reads, and no read is waiting when
        // only a reduction is.
        bool busy() const noexcept { return !pending_.empty(); }

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
            std::uint64_t key = 0;
            std::vector<ReducedVolume> channels;
            long long micros = 0;
        };
        // Queues the reduction of `channels` into 3D bricks; a newer request
        // replaces an older one that has not started.
        void reduce(std::uint64_t key, std::vector<Channel> channels);
        std::uint64_t reductionKey() const noexcept { return reductionKey_; }

        // Forgets every queued and running job: results that still arrive are
        // dropped. Called when the displayed output changes.
        void cancelAll();

        // --- results (called on the GUI thread) ----------------------------------
        std::function<void(const Volume&)> volumeReady;
        std::function<void(const Reduction&)> reductionReady;
        std::function<void(double fraction, const std::string& message)> volumeProgress;

    private:
        struct Job {
            const StepOutput* out = nullptr;
            Index c = 0, t = 0;
            bool operator<(const Job& o) const noexcept {
                return out != o.out ? out < o.out : (c != o.c ? c < o.c : t < o.t);
            }
        };

        // Set by retain() (on the GUI thread) for a read no longer wanted.
        using Dropped = std::shared_ptr<std::atomic<bool>>;
        struct Queued {
            std::function<void()> run;
            Dropped dropped;   // a volume read's flag; null for a reduction
        };

        void enqueue(Dropped dropped, std::function<void()> job);
        void run();

        Poster post_;
        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<Queued> queue_;
        bool quit_ = false;                                        // guarded by mutex_

        std::shared_ptr<std::atomic<std::uint64_t>> generation_;   // shared with the jobs
        std::shared_ptr<std::atomic<std::uint64_t>> latestReduction_;   // the key of the newest reduce()
        std::shared_ptr<std::atomic<bool>> alive_;                 // cleared by the destructor
        std::uint64_t gen_ = 1;
        std::map<Job, Dropped> pending_;   // reads queued or running, with their flags (GUI thread)
        bool reductionPending_ = false;
        std::uint64_t reductionKey_ = 0;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_LOADER_HPP
