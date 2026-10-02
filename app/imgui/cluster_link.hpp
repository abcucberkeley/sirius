#ifndef SIRIUS_IMGUI_CLUSTER_LINK_HPP
#define SIRIUS_IMGUI_CLUSTER_LINK_HPP

// The window's side of a cluster session (core/cluster.hpp): one per
// application, owned by App.
//
//   * the profile, remembered in the settings ("cluster/profile");
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

        cluster::Profile storedProfile() const;
        void connect(const cluster::Profile& profile);   // saves the profile
        // Off the GUI thread (scancel takes a moment); `cancelJob` scancels.
        void disconnect(bool cancelJob);
        // Asks whether to cancel the job too (default yes), then disconnects.
        void disconnectAsking();
        // The HPC endpoint while connected (through the tunnel); empty host otherwise.
        RemoteConfig remoteConfig() const;

        // Once a frame: state changes, the pending prompt.
        void frame();

        // "HPC: n0123 · connected", its colour; "" while there was never a session.
        std::string indicator(ImU32& color) const;

        // The cluster's folders opened last, newest first.
        std::vector<std::string> recentFolders() const;
        void addRecentFolder(const std::string& path);

        // Quitting: a job still running is the user's to keep or cancel.
        // Calls `done` (GUI thread) once settled; at once when there is no job.
        void settleBeforeQuit(std::function<void()> done);

    private:
        std::optional<std::string> ask(const ssh::Prompt& p);
        void onState(const cluster::Status& st);

        App& app_;
        cluster::Session session_;
        std::shared_ptr<RemoteDatasets> datasets_;
        cluster::State lastState_ = cluster::State::Idle;
        std::shared_ptr<std::atomic<bool>> alive_;
        std::thread disconnecting_;
    };

    // The modal box for one ssh prompt (dialogs/cluster_dialog.cpp).
    std::shared_ptr<Dialog> makeClusterPromptDialog(App& app, std::shared_ptr<ClusterPrompt> prompt);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_CLUSTER_LINK_HPP
