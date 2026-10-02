#ifndef SIRIUS_IMGUI_HTTP_HPP
#define SIRIUS_IMGUI_HTTP_HPP

// HTTP(S) for the assistant (OpenAI-compatible chat endpoints, streamed) and
// the model hub (Hugging Face search and downloads), over libcurl with the
// platform's TLS.
//
// Two levels:
//   * perform() / download(): one blocking request on the calling thread,
//     for code that already runs on a worker thread;
//   * Fetch: a request on a thread of its own whose callbacks arrive on the
//     GUI thread (through the poster the application installs), for panels.
//     Destroying or restarting a Fetch cancels the request in flight and its
//     callbacks are never called afterwards.

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::app::gui::http {

    struct Request {
        std::string method = "GET";
        std::string url;
        std::vector<std::pair<std::string, std::string>> headers;
        std::string body;
        long connectTimeoutSeconds = 15;
        long timeoutSeconds = 0;               // whole transfer; 0 = none
        // Give up when fewer than 1 byte/s arrive for this long; 0 = never.
        // (A streamed answer may be silent for minutes while a model loads.)
        long stallSeconds = 0;
        bool followRedirects = true;
        // Bearer token, sent as "Authorization: Bearer <token>" when non-empty,
        // and never forwarded to another host by a redirect, nor from https
        // to http. A request with one to an http:// address other than this
        // machine's is refused (Response::error says so). Only http(s) URLs
        // are fetched at all.
        std::string bearer;
    };

    struct Response {
        long status = 0;                       // 0 when the request never got an answer
        std::string body;                      // empty when the data went to onData / a file
        std::string error;                     // transport error ("" when an answer arrived)
        bool cancelled = false;
        std::map<std::string, std::string> headers;   // names in lower case
        bool ok() const noexcept { return error.empty() && !cancelled && status >= 200 && status < 300; }
        // The error, else "HTTP <status>".
        std::string message() const;
    };

    struct Callbacks {
        // Every chunk of the body as it arrives; the body is then not kept in
        // the Response. Return false to abort.
        std::function<bool(const char* data, std::size_t size)> onData;
        // Bytes received so far and expected in all (0 when unknown).
        std::function<void(double received, double total)> onProgress;
        // Checked while the transfer runs: true aborts it.
        const std::atomic<bool>* cancel = nullptr;
    };

    // Blocking.
    Response perform(const Request& request, const Callbacks& callbacks = {});
    // The body into `path` (written beside it as <path>.part and renamed when
    // complete; a failed download leaves no file).
    Response download(const Request& request, const std::string& path, const Callbacks& callbacks = {});

    std::string urlEncode(const std::string& text);
    // "https://host:port" of a URL ("" when it has no scheme).
    std::string origin(const std::string& url);

    // How results reach the GUI thread: the application installs a function
    // that queues `fn` for the start of the next frame (Bridge::post).
    void setGuiPoster(std::function<void(std::function<void()>)> poster);
    void postToGui(std::function<void()> fn);

    // One request at a time on its own thread, callbacks on the GUI thread.
    class Fetch {
    public:
        struct Handlers {
            // Body chunks, in order, as they arrive (streaming). When set the
            // Response passed to `done` has an empty body.
            std::function<void(const std::string& chunk)> onData;
            std::function<void(double received, double total)> onProgress;
            std::function<void(const Response& response)> done;
        };

        Fetch();
        ~Fetch();   // cancels
        Fetch(const Fetch&) = delete;
        Fetch& operator=(const Fetch&) = delete;

        // Cancels what is in flight and starts `request`. With `path` the body
        // goes to that file (download()).
        void start(const Request& request, Handlers handlers, const std::string& path = {});
        // No callback of the request in flight is called after this returns.
        void cancel();
        bool busy() const noexcept;

    private:
        struct State;
        std::shared_ptr<State> state_;
        std::thread thread_;
    };

} // namespace sirius::app::gui::http

#endif // SIRIUS_IMGUI_HTTP_HPP
