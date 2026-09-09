#include "core/session_log.hpp"

#include <ctime>
#include <stdexcept>

namespace sirius::app {

    SessionLog::~SessionLog() { stop(); }

    void SessionLog::start(const std::filesystem::path& path, nlohmann::json header) {
        std::lock_guard<std::mutex> g(mutex_);
        if (out_.is_open()) {
            // the previous recording ends the way stop() ends it
            write({{"event", "stopped"}, {"events", lines_}});
            out_.flush();
            out_.close();
        }
        std::error_code ec;
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
        out_.open(path, std::ios::out | std::ios::app);
        if (!out_) throw std::runtime_error("cannot write the session recording: " + path.string());
        path_ = path;
        began_ = std::chrono::steady_clock::now();
        lines_ = 0;

        if (!header.is_object()) header = nlohmann::json::object();
        header["event"] = "session";
        header["t"] = 0.0;
        // wall clock too: the elapsed times are relative, and a reader will
        // want to line a recording up against the files it produced
        const std::time_t now = std::time(nullptr);
        char stamp[32] = {0};
        std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));
        header["started"] = stamp;
        header["version"] = 1;
        out_ << header.dump() << "\n";
        out_.flush();
        ++lines_;
    }

    void SessionLog::stop() {
        std::lock_guard<std::mutex> g(mutex_);
        if (!out_.is_open()) return;
        // written here rather than by the caller so that a recording closed by
        // the application quitting still ends with a terminator
        write({{"event", "stopped"}, {"events", lines_}});
        out_.flush();
        out_.close();
        path_.clear();
    }

    bool SessionLog::recording() const {
        std::lock_guard<std::mutex> g(mutex_);
        return out_.is_open();
    }

    std::uint64_t SessionLog::lines() const {
        std::lock_guard<std::mutex> g(mutex_);
        return lines_;
    }

    void SessionLog::record(std::string event, nlohmann::json fields) {
        std::lock_guard<std::mutex> g(mutex_);
        if (!out_.is_open()) return;
        if (!fields.is_object()) fields = nlohmann::json{{"value", std::move(fields)}};
        fields["event"] = std::move(event);
        write(std::move(fields));
    }

    // caller holds the mutex and has checked that the file is open
    void SessionLog::write(nlohmann::json line) {
        line["t"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - began_).count();
        out_ << line.dump() << "\n";
        // flushed per line: a recording is worth having even if the
        // application is killed mid-session
        out_.flush();
        ++lines_;
    }

} // namespace sirius::app
