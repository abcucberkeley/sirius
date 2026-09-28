#include "imgui/http.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <system_error>

#include <curl/curl.h>

#include "imgui/strings.hpp"

namespace sirius::app::gui::http {

    namespace {

        void ensureCurl() {
            static const bool once = [] {
                curl_global_init(CURL_GLOBAL_DEFAULT);
                return true;
            }();
            (void)once;
        }

        struct Transfer {
            const Callbacks* callbacks = nullptr;
            Response* response = nullptr;
            std::ofstream* file = nullptr;
            bool aborted = false;
        };

        std::size_t onWrite(char* data, std::size_t size, std::size_t count, void* user) {
            auto* t = static_cast<Transfer*>(user);
            const std::size_t n = size * count;
            if (t->callbacks->cancel && t->callbacks->cancel->load()) {
                t->aborted = true;
                return 0;
            }
            if (t->file) {
                t->file->write(data, static_cast<std::streamsize>(n));
                if (!*t->file) return 0;
            } else if (t->callbacks->onData) {
                if (!t->callbacks->onData(data, n)) {
                    t->aborted = true;
                    return 0;
                }
            } else {
                t->response->body.append(data, n);
            }
            return n;
        }

        std::size_t onHeader(char* data, std::size_t size, std::size_t count, void* user) {
            auto* t = static_cast<Transfer*>(user);
            const std::size_t n = size * count;
            const std::string line(data, n);
            if (startsWith(line, "HTTP/")) {
                t->response->headers.clear();   // the headers of the answer after a redirect
                return n;
            }
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos)
                t->response->headers[toLower(trimmed(line.substr(0, colon)))] = trimmed(line.substr(colon + 1));
            return n;
        }

        int onProgress(void* user, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
            auto* t = static_cast<Transfer*>(user);
            if (t->callbacks->cancel && t->callbacks->cancel->load()) {
                t->aborted = true;
                return 1;
            }
            if (t->callbacks->onProgress) t->callbacks->onProgress(static_cast<double>(now), static_cast<double>(total));
            return 0;
        }

        Response run(const Request& request, const Callbacks& callbacks, std::ofstream* file) {
            ensureCurl();
            Response response;
            CURL* curl = curl_easy_init();
            if (!curl) {
                response.error = "cannot initialise libcurl";
                return response;
            }
            Transfer transfer;
            transfer.callbacks = &callbacks;
            transfer.response = &response;
            transfer.file = file;

            curl_slist* headers = nullptr;
            for (const auto& h : request.headers) headers = curl_slist_append(headers, (h.first + ": " + h.second).c_str());
            char errorBuffer[CURL_ERROR_SIZE] = {0};

            curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
            curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer);
            curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "sirius-imgui/" SIRIUS_VERSION);
            curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, request.followRedirects ? 1L : 0L);
            curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, request.connectTimeoutSeconds);
            if (request.timeoutSeconds > 0) curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.timeoutSeconds);
            if (request.stallSeconds > 0) {
                curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
                curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, request.stallSeconds);
            }
            if (!request.bearer.empty()) {
                // as a credential rather than a header: libcurl drops it when a
                // redirect leaves the host it was meant for
                curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
                curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, request.bearer.c_str());
            }
            if (request.method == "POST") {
                curl_easy_setopt(curl, CURLOPT_POST, 1L);
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.body.data());
                curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body.size()));
            } else if (request.method == "HEAD") {
                curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
            } else if (request.method != "GET") {
                curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, request.method.c_str());
                if (!request.body.empty()) {
                    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.body.data());
                    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body.size()));
                }
            }
            if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, onWrite);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, onHeader);
            curl_easy_setopt(curl, CURLOPT_HEADERDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, onProgress);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

            const CURLcode code = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status);
            const bool cancelled = transfer.aborted || (callbacks.cancel && callbacks.cancel->load());
            if (cancelled) {
                response.cancelled = true;
            } else if (code != CURLE_OK) {
                response.error = errorBuffer[0] ? std::string(errorBuffer) : std::string(curl_easy_strerror(code));
            }
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return response;
        }

        std::function<void(std::function<void()>)>& poster() {
            static std::function<void(std::function<void()>)> p;
            return p;
        }
        std::mutex& posterMutex() {
            static std::mutex m;
            return m;
        }

    } // namespace

    std::string Response::message() const {
        if (cancelled) return "cancelled";
        if (!error.empty()) return error;
        return "HTTP " + std::to_string(status);
    }

    Response perform(const Request& request, const Callbacks& callbacks) { return run(request, callbacks, nullptr); }

    Response download(const Request& request, const std::string& path, const Callbacks& callbacks) {
        const std::filesystem::path target = std::filesystem::u8path(path);
        std::filesystem::path part = target;
        part += ".part";
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        Response response;
        {
            std::ofstream file(part, std::ios::binary | std::ios::trunc);
            if (!file) {
                response.error = "cannot write " + part.u8string();
                return response;
            }
            response = run(request, callbacks, &file);
            file.flush();
            if (!file && response.error.empty() && !response.cancelled) response.error = "cannot write " + part.u8string();
        }
        if (!response.ok()) {
            // what arrived is the server's error message, not the file
            if (response.status >= 400) {
                std::ifstream f(part, std::ios::binary);
                std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                if (text.size() > 4096) text.resize(4096);
                response.body = text;
            }
            std::filesystem::remove(part, ec);
            return response;
        }
        std::filesystem::remove(target, ec);
        ec.clear();
        std::filesystem::rename(part, target, ec);
        if (ec) {
            response.error = "cannot move the download to " + path + ": " + ec.message();
            std::filesystem::remove(part, ec);
        }
        return response;
    }

    std::string urlEncode(const std::string& text) {
        std::string out;
        char buf[4];
        for (unsigned char c : text) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                out += static_cast<char>(c);
            } else {
                std::snprintf(buf, sizeof buf, "%%%02X", c);
                out += buf;
            }
        }
        return out;
    }

    std::string origin(const std::string& url) {
        const std::size_t scheme = url.find("://");
        if (scheme == std::string::npos) return std::string();
        const std::size_t end = url.find_first_of("/?#", scheme + 3);
        return end == std::string::npos ? url : url.substr(0, end);
    }

    void setGuiPoster(std::function<void(std::function<void()>)> p) {
        const std::lock_guard<std::mutex> g(posterMutex());
        poster() = std::move(p);
    }

    void postToGui(std::function<void()> fn) {
        std::function<void(std::function<void()>)> p;
        {
            const std::lock_guard<std::mutex> g(posterMutex());
            p = poster();
        }
        if (p) p(std::move(fn));
    }

    // --- Fetch ------------------------------------------------------------------

    struct Fetch::State {
        std::atomic<bool> cancel{false};
        std::atomic<bool> busy{false};
    };

    Fetch::Fetch() : state_(std::make_shared<State>()) {}

    Fetch::~Fetch() { cancel(); }

    bool Fetch::busy() const noexcept { return state_->busy.load(); }

    void Fetch::cancel() {
        state_->cancel.store(true);
        if (thread_.joinable()) thread_.join();
        // the callbacks already posted to the GUI thread look at the state
        // they were posted with, which stays cancelled; the next request gets
        // a state of its own
        state_ = std::make_shared<State>();
    }

    void Fetch::start(const Request& request, Handlers handlers, const std::string& path) {
        cancel();
        const std::shared_ptr<State> state = state_;
        state->busy.store(true);
        thread_ = std::thread([state, request, handlers = std::move(handlers), path] {
            Callbacks cb;
            cb.cancel = &state->cancel;
            if (handlers.onData && path.empty()) {
                cb.onData = [&](const char* data, std::size_t size) {
                    std::string chunk(data, size);
                    postToGui([state, fn = handlers.onData, chunk = std::move(chunk)] {
                        if (!state->cancel.load()) fn(chunk);
                    });
                    return !state->cancel.load();
                };
            }
            auto lastProgress = std::chrono::steady_clock::now() - std::chrono::seconds(1);
            if (handlers.onProgress) {
                cb.onProgress = [&](double received, double total) {
                    // libcurl reports many times a second: ten are plenty for a bar
                    const auto now = std::chrono::steady_clock::now();
                    if (now - lastProgress < std::chrono::milliseconds(100) && received < total) return;
                    lastProgress = now;
                    postToGui([state, fn = handlers.onProgress, received, total] {
                        if (!state->cancel.load()) fn(received, total);
                    });
                };
            }
            Response response = path.empty() ? perform(request, cb) : download(request, path, cb);
            postToGui([state, fn = handlers.done, response = std::move(response)] {
                state->busy.store(false);
                if (!state->cancel.load() && fn) fn(response);
            });
        });
    }

} // namespace sirius::app::gui::http
