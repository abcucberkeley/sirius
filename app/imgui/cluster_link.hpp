#ifndef SIRIUS_IMGUI_CLUSTER_LINK_HPP
#define SIRIUS_IMGUI_CLUSTER_LINK_HPP

// The window's side of a cluster session (core/cluster.hpp): one per
// application, owned by App.
//
//   * the cluster profiles, kept in the settings file ([cluster.<name>],
//     core/cluster_profiles.hpp; the single profile of before migrated once);
//   * ssh's prompts: the session's askpass relay asks on a thread of its
//     own, and this shows the prompt in a modal password box on the GUI
//     thread (the answer is handed to ssh and forgotten: never stored,
//     never logged);
//   * what follows a state change: connected -> the HPC backend points at
//     the worker through the tunnel and cluster datasets open
//     (core/remote_source.hpp); disconnected -> both go away; a log line for
//     each;
//   * the cluster's recent folders, for the browser.
//
// GUI thread, except where a function says otherwise.

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <imgui.h>

#include "core/cluster.hpp"
#include "core/cluster_profiles.hpp"
#include "core/remote_source.hpp"
#include "core/workbench.hpp"

namespace sirius::app::gui {

    class App;
    class Dialog;

    // A prompt ssh is waiting on, shared by the asking thread and the box.
    struct ClusterPrompt {
        ssh::Prompt prompt;
        std::string host;
        std::atomic<bool> answered{false};
        std::atomic<bool> abandoned{false};   // the login ended meanwhile: the box closes itself
        bool cancelled = true;
        std::string answer;                   // read once by the asking thread, then wiped
    };

    class ClusterLink {
    public:
        explicit ClusterLink(App& app);
        ~ClusterLink();
        ClusterLink(const ClusterLink&) = delete;
        ClusterLink& operator=(const ClusterLink&) = delete;

        cluster::Session& session() noexcept { return session_; }
        cluster::Status status() const { return session_.status(); }
        bool connected() const { return session_.connected(); }
        bool sshUp() const { return session_.sshUp(); }

        // The profiles as the settings file has them (the single profile of
        // before migrated into it the first time), and the one in use.
        cluster::ProfileBook profiles() const;
        // Writes `book` to the settings: its profiles, the current one, and
        // the removal of profiles it no longer has.
        void saveProfiles(const cluster::ProfileBook& book);
        // `profile` stored in the book (under its name) as the current one.
        void saveProfile(const cluster::Profile& profile);
        cluster::Profile storedProfile() const;

        // Step 1, the job: the SSH login, then a job that holds the
        // allocation (saves the profile).
        void connectJob(const cluster::Profile& profile);
        // Step 2, the worker in that job (saves the profile); restarts a
        // running worker in the same job.
        void startWorker(const cluster::Profile& profile);
        // Both steps (scripted runs).
        void connect(const cluster::Profile& profile);
        // The SSH login and the cluster's partitions only, no job; saves the profile.
        void logIn(const cluster::Profile& profile);
        // Off the GUI thread: the worker step ends, the job stays.
        void stopWorker();
        // Off the GUI thread: the job cancelled, then a new one with `profile`
        // (and its worker, when one ran). Asks first.
        void newJobAsking(const cluster::Profile& profile);
        // Builds a worker image in the held job (the cluster's paths).
        void buildImage(const cluster::Profile& profile, const std::string& defFile, const std::string& image);
        // Off the GUI thread (scancel takes a moment); `cancelJob` scancels.
        void disconnect(bool cancelJob);
        // Asks whether to cancel the job too (default yes), then disconnects.
        void disconnectAsking();
        // The HPC endpoint while connected (through the tunnel); empty host otherwise.
        RemoteConfig remoteConfig() const;

        // Once a frame: state changes, the pending prompt.
        void frame();

        // "HPC: n0123 · GPU" (the session's HPC device), its colour; "" while
        // there was never a session.
        std::string indicator(ImU32& color) const;
        // The title bar's button (cluster::connectionBadge with the session's device).
        cluster::ConnectionBadge badge() const;

        // Whether the HPC device's GPU can be chosen; else `why` says what
        // stops it: the connected worker reports no CUDA, or the profile
        // asks for no GPU. Without a session the stored profile decides.
        bool hpcGpuUsable(std::string* why = nullptr) const;
        // What the connected node computes on, its GPU then its CPU
        // (cluster::nodeDevices); empty while no worker is connected.
        std::vector<cluster::NodeDevice> nodeDevices() const;

        // The cluster's folders opened last, newest first.
        std::vector<std::string> recentFolders() const;
        void addRecentFolder(const std::string& path);

        // Quitting: a job still running is the user's to keep or cancel.
        // Calls `done` (GUI thread) once settled; at once when there is no job.
        void settleBeforeQuit(std::function<void()> done);

    private:
        // `profile` saved (its Slurm choice remembered for its host), and
        // the fake ssh of tests and screenshots put in for ssh.
        cluster::Profile prepared(const cluster::Profile& profile);
        std::optional<std::string> ask(const ssh::Prompt& p);
        void onState(const cluster::Status& st);

        App& app_;
        cluster::Session session_;
        std::shared_ptr<RemoteDatasets> datasets_;
        cluster::State lastState_ = cluster::State::Idle;
        std::shared_ptr<std::atomic<bool>> alive_;
        std::thread disconnecting_;   // a disconnect, stop or new job, off the GUI thread
        // Runs `fn` on disconnecting_ (after the one before it).
        void offThread(std::function<void()> fn);
    };

    // The modal box for one ssh prompt (dialogs/cluster_dialog.cpp).
    std::shared_ptr<Dialog> makeClusterPromptDialog(App& app, std::shared_ptr<ClusterPrompt> prompt);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_CLUSTER_LINK_HPP
