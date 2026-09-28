#ifndef SIRIUS_IMGUI_VIEWER_TRACE_HPP
#define SIRIUS_IMGUI_VIEWER_TRACE_HPP

// SIRIUS_TRACE_VIEW=1 prints what the hot UI paths cost. ScopedTrace starts
// a timer when it is constructed and logs "<what> N us" when it leaves the
// scope; with the variable unset it does nothing at all, so a trace can sit
// in a paint or drag path.
//
//     ScopedTrace trace("layoutPanes");

#include <chrono>
#include <cstdio>
#include <string>

#include "imgui/platform.hpp"

namespace sirius::app::gui {

    // A stopwatch in microseconds.
    class TraceClock {
    public:
        void start() { from_ = std::chrono::steady_clock::now(); }
        long long micros() const {
            return static_cast<long long>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - from_).count());
        }
        long long restart() {
            const long long us = micros();
            start();
            return us;
        }

    private:
        std::chrono::steady_clock::time_point from_ = std::chrono::steady_clock::now();
    };

    // One line on stderr, where the scripting output goes too.
    inline void traceLine(const std::string& line) {
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    }

    class ScopedTrace {
    public:
        static bool enabled() {
            static const bool on = platform::hasEnvironment("SIRIUS_TRACE_VIEW");
            return on;
        }

        explicit ScopedTrace(const char* what) : what_(what), on_(enabled()) {
            if (on_) clock_.start();
        }
        ~ScopedTrace() {
            if (on_) std::fprintf(stderr, "%s %lld us\n", what_, clock_.micros());
        }

        ScopedTrace(const ScopedTrace&) = delete;
        ScopedTrace& operator=(const ScopedTrace&) = delete;

    private:
        const char* what_;
        bool on_;
        TraceClock clock_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_TRACE_HPP
