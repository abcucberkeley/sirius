#ifndef SIRIUS_APP_CLUSTER_WIZARD_HPP
#define SIRIUS_APP_CLUSTER_WIZARD_HPP

// What the cluster dialog's three pages decide, without the GUI: which page
// a session opens on, when Next (and Finish) may be pressed and why not,
// the login's outcome in plain words, the job's line, and the worker's
// health report from its hello (core/cluster.hpp's Status).
//
//   1 Connect   the SSH login: Next once logged in to the field's host
//   2 Job       the job that holds a node: Next once Slurm gave it one
//   3 Worker    the worker in the job: Finish once SIRIUS's engine answers
//               (a worker without the engine is not ready: nothing would run)
//   Summary     a session set up already: where it runs, how it is
//
// GUI-free: the tests check it as it is.

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "core/build_info.hpp"
#include "core/cluster.hpp"

namespace sirius::app::cluster::wizard {

    enum class Page { Connect,
                      Job,
                      Worker,
                      Summary };

    // Logged in to `host` (the Cluster field, trimmed): the SSH session is up for it.
    bool loggedIn(const Status& st, const std::string& host);
    // A job holds a node for this session (JobReady, Starting or Connected).
    bool jobHeld(const Status& st);
    // Logging in, getting the job or starting the worker.
    bool busy(const Status& st);

    // A button's state: whether it may be pressed, and when not, why (a tooltip).
    struct Gate {
        bool enabled = false;
        std::string why;
    };
    // Next on `page` (Finish on the Worker page; the Summary has none).
    Gate nextGate(Page page, const Status& st, const std::string& host);
    // Connect on page 1: a host typed, nothing running, no job held (which
    // ties the session to its host until disconnected).
    Gate connectGate(const Status& st, const std::string& host);
    // Start job on page 2: logged in to `host`, no job held, nothing running.
    Gate startJobGate(const Status& st, const std::string& host);
    // Start worker on page 3: a job held, nothing running, an image set.
    Gate startWorkerGate(const Status& st, const std::string& image);
    // A page's tick in the header: what it is for is done.
    bool pageDone(Page page, const Status& st, const std::string& host);
    // The page the dialog opens on: the Summary once a worker answers, the
    // page of what is under way or failed, else the first.
    Page openingPage(const Status& st, const std::string& host);

    // --- page 1: the login's outcome ---------------------------------------------------
    struct LoginOutcome {
        enum class Kind { None,      // not tried yet (or for another host)
                          Busy,      // logging in
                          Ok,        // "Connected to fiona as velat"
                          Failed };  // what went wrong, in plain words
        Kind kind = Kind::None;
        std::string text;
        std::string details;          // ssh's own output ("Details")
    };
    // `user`: the login's user as the cluster said it ("" unknown).
    LoginOutcome loginOutcome(const Status& st, const std::string& host, const std::string& user);
    // ssh's failure in plain words: a wrong password, a host not found, a
    // timeout, a refused connection, a host key that changed... (`sshOutput`
    // is ssh's stderr; `reason` the session's words for it).
    std::string loginFailureWords(const std::string& host, const std::string& reason, const std::string& sshOutput);

    // --- page 2: the job ----------------------------------------------------------------
    struct JobLine {
        enum class Kind { None,       // no job asked for yet
                          Busy,       // submitting, queued
                          Running,    // the job holds its node
                          Failed };
        Kind kind = Kind::None;
        std::string text;             // "Queued · Resources · waited 0:42", "Running on g0003 · job 4711 · 50 min left"
    };
    JobLine jobLine(const Status& st, std::chrono::steady_clock::time_point now);
    // "50 min left" ("" when the limit or the start is unknown; "no time limit").
    std::string timeLeftText(const Status& st, std::chrono::steady_clock::time_point now);

    // --- page 3: the worker's health ------------------------------------------------------
    enum class Mark { Ok,
                      Warn,
                      Fail,
                      Info };
    struct HealthRow {
        std::string label;            // "GPU", "CUDA", "torch", ...
        std::string value;
        Mark mark = Mark::Info;
    };
    struct HealthReport {
        enum class Verdict { None,     // no worker started yet
                             Busy,     // starting
                             Ready,    // the worker answers
                             Failed }; // it did not start, or stopped
        Verdict verdict = Verdict::None;
        std::string headline;          // "Ready", "Not ready: ..."
        std::string fix;               // what to do (Failed)
        std::string details;           // the cluster's own output (Failed)
        std::vector<HealthRow> rows;
    };
    // From the status (the hello's capabilities once connected), the profile
    // the worker was started with and this application's build.
    HealthReport healthReport(const Status& st, const Profile& p, const BuildInfo& app, std::chrono::steady_clock::time_point now);

    // --- the HPC backend without an engine -------------------------------------------------
    // SIRIUS's engine answers in this session: the HPC backend may run.
    bool engineReady(const Status& st);
    // Why the HPC backend can run nothing now, one line for the Run buttons'
    // tooltips, the panels and the status bar ("HPC: no SIRIUS engine on the
    // cluster \xE2\x80\x94 open Cluster to fix"); "" when the engine answers.
    std::string hpcNoEngineReason(const Status& st);

} // namespace sirius::app::cluster::wizard

#endif // SIRIUS_APP_CLUSTER_WIZARD_HPP
