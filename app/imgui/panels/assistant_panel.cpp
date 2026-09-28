#include "imgui/panels/assistant_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>
#include <nlohmann/json.hpp>

#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/panels/assistant_markdown.hpp"
#include "imgui/panels/llm_client.hpp"
#include "imgui/platform.hpp"
#include "imgui/secret_store.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

namespace sirius::app::gui {

    // --- settings ------------------------------------------------------------------

    AssistantSettings AssistantSettings::load() {
        const Settings& s = gui::settings();
        AssistantSettings a;
        a.provider = s.getString("assistant/provider", a.provider);
        a.baseUrl = s.getString("assistant/baseUrl", a.baseUrl);
        a.model = s.getString("assistant/model", a.model);
        a.apiKey = secrets::read("assistant/apiKey");
        a.askBeforeActing = s.getBool("assistant/askBeforeActing", a.askBeforeActing);
        if (a.apiKey.empty()) a.apiKey = environmentKey(a.provider, &a.apiKeyVariable);
        return a;
    }

    void AssistantSettings::save() const {
        Settings& s = gui::settings();
        s.set("assistant/provider", provider);
        s.set("assistant/baseUrl", baseUrl);
        s.set("assistant/model", model);
        s.set("assistant/askBeforeActing", askBeforeActing);
    }

    std::string AssistantSettings::requestKey() const { return provider == "ollama" ? std::string() : apiKey; }

    bool AssistantSettings::storeApiKey(const std::string& key) { return secrets::write("assistant/apiKey", key); }

    std::string AssistantSettings::environmentKey(const std::string& provider, std::string* variable) {
        std::vector<const char*> names;
        if (provider == "openrouter") names.push_back("OPENROUTER_API_KEY");
        names.push_back("SIRIUS_LLM_API_KEY");
        for (const char* name : names) {
            const std::string value = platform::environment(name);
            if (value.empty()) continue;
            if (variable) *variable = name;
            return value;
        }
        if (variable) variable->clear();
        return std::string();
    }

    namespace {

        using json = nlohmann::json;
        using theme::px;
        using Clock = std::chrono::steady_clock;
        namespace md = assistant_markdown;

        // Text from a model may hold bytes that are not UTF-8: replaced rather than thrown over.
        std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

        bool mutatingTool(const std::string& name) {
            static const char* readOnly[] = {"get_", "list_", "read_", "describe", "help", "explain", "context", "find_"};
            for (const char* prefix : readOnly)
                if (startsWith(name, prefix)) return false;
            return true;
        }

        // The first `n` characters (code points) of `s`.
        std::string leftChars(const std::string& s, std::size_t n, bool* cut = nullptr) {
            std::size_t i = 0, count = 0;
            while (i < s.size() && count < n) {
                nextCodepoint(s, i);
                ++count;
            }
            if (cut) *cut = i < s.size();
            return s.substr(0, i);
        }

        float fontPx(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return designPx * st.FontScaleMain * st.FontScaleDpi;
        }

        float lineHeight(float designPx, theme::Weight w = theme::Weight::Regular) { return theme::textSize("Ag", designPx, w).y; }

        void place(float x, float y) { ImGui::SetCursorScreenPos(ImVec2(x, y)); }

        // Tells the layout where hand-placed content ended.
        void reach(float x, float y) {
            place(x, y);
            ImGui::Dummy(ImVec2(0.0f, 0.0f));
        }

        // Design pixels from display pixels (what widgets:: take for widths).
        float dp(float display) { return display / std::max(theme::scale(), 0.01f); }

        // The size a chip button takes (widgets::button's Chip metrics), display pixels.
        ImVec2 chipSize(const std::string& label) {
            const ImVec2 ts = theme::textSize(label, 11);
            return ImVec2(theme::snap(ts.x + 2 * px(8) + 2 * px(theme::kBorder)), theme::snap(ts.y + 2 * px(3) + 2 * px(theme::kBorder)));
        }

        // Tool rounds one question may take before the loop stops and asks the
        // user: a model that keeps calling tools without ever answering would
        // otherwise go on for as long as the server lets it.
        constexpr int kMaxRounds = 25;

        // What a tool result may weigh in the conversation.
        constexpr std::size_t kMaxToolResult = 12000;

    } // namespace

    // --- panel -----------------------------------------------------------------------

    struct AssistantPanel::Impl {
        App& app;
        AssistantSettings settings;
        LlmClient client;
        // Callbacks deferred to between two frames look here first: the
        // panel may be gone by the time they run.
        std::shared_ptr<int> alive = std::make_shared<int>(0);

        // --- the transcript ----------------------------------------------------
        struct PendingCall {
            std::string id, name, arguments;
        };
        // What sits below an assistant text: an action card, or the question
        // whether to apply a call.
        struct Attachment {
            enum class Kind { Card,
                              Confirm };
            Kind kind = Kind::Card;
            ActionRecord rec;           // Card
            PendingCall call;           // Confirm
            bool answered = false;      // Confirm: gone once answered
        };
        struct Block {
            bool user = false;
            std::string text;           // as typed / the Markdown so far
            bool showText = true;
            std::vector<Attachment> below;
            // the text laid out for a width at a scale
            md::Layout layout;
            bool dirty = true;
        };
        std::vector<std::unique_ptr<Block>> blocks;
        Block* streamingBlock = nullptr;   // the block the reply being received goes to
        bool streamingLabel = false;       // ... and its text is the one streamed
        std::string streamingSource;
        int scrollDown = 0;                // frames left to hold the transcript at its end
        std::string copyText;              // what the context menu copies

        // --- busy -----------------------------------------------------------------
        // What the busy row says while a reply is pending: the base text,
        // then the seconds, then why it may be taking long (a 20 GB model
        // took two minutes to load on first use, with nothing else to show).
        bool busy = false;
        std::string busyBase;
        Clock::time_point busySince = Clock::now();
        int reasoningChars = 0;
        bool waitingForModel = false;   // between the request and its first byte

        // --- the conversation ---------------------------------------------------
        json history = json::array();   // API messages after the system prompt
        std::deque<PendingCall> pending;
        int rounds = 0;                  // tool rounds of the question being answered
        // Moves when the user stops the assistant: whatever was on its way for
        // the conversation before (a deferred call, a model list) is dropped.
        int generation = 0;
        bool executing = false;          // a tool call is running (a run pumps frames meanwhile)
        bool stopAfterCall = false;      // stopped while it did
        int runStartedSlot = 0;

        // --- the footer ----------------------------------------------------------
        std::string input;
        bool focusRequest = false;
        bool refocusWhenIdle = false;
        std::string modelText;            // the dropdown's text
        std::vector<std::string> modelItems;
        std::vector<std::string> listedModels;   // what the server said it has, empty until it answers
        std::string modelTip = "The model that answers: the server's list, or a name typed here";
        bool modelEditing = false;        // the dropdown's text field had the keyboard last frame
        int modelLookup = 0;              // the latest model list asked for

        explicit Impl(App& a) : app(a) {
            settings = AssistantSettings::load();
            client.onDelta = [this](const std::string& t) { onDelta(t); };
            client.onThinking = [this](int chars) {
                waitingForModel = false;
                reasoningChars = chars;
            };
            client.onFinished = [this](const json& m) { onFinished(m); };
            client.onFailed = [this](const std::string& e) {
                pending.clear();
                showError(format("The model server answered: %s (provider %s, %s)", e.c_str(), settings.provider.c_str(),
                                 settings.baseUrl.c_str()));
            };
            // A run a tool asked for pumps frames until it is done (the run
            // hook in main); the busy row says what the wait is.
            runStartedSlot = app.bridge().runStarted.connect([this] {
                if (executing && !stopAfterCall) setBusy(true, "Running the pipeline…");
            });
            refreshModelBox();
            addAssistantBlock("Ask about a step, or tell me what to do: I can edit parameters, run steps and change the view. "
                              "Every change lands in the undo stack.");
        }

        ~Impl() { app.bridge().runStarted.disconnect(runStartedSlot); }

        // Runs `fn` between two frames, unless the panel is gone or the
        // conversation was stopped meanwhile.
        void later(std::function<void()> fn) {
            std::weak_ptr<int> token = alive;
            const int gen = generation;
            app.defer([this, token, gen, fn = std::move(fn)] {
                if (token.expired() || gen != generation) return;
                fn();
            });
        }

        // --- transcript --------------------------------------------------------
        void scrollToBottom() {
            scrollDown = 3;
            app.requestRedraw();
        }

        void addUserBubble(const std::string& text) {
            auto b = std::make_unique<Block>();
            b->user = true;
            b->text = text;
            blocks.push_back(std::move(b));
            scrollToBottom();
        }

        // A block: assistant text + cards below it.
        Block* addAssistantBlock(const std::string& text) {
            auto b = std::make_unique<Block>();
            b->text = text;
            b->showText = !text.empty();
            Block* block = b.get();
            blocks.push_back(std::move(b));
            streamingBlock = block;
            streamingLabel = true;
            scrollToBottom();
            return block;
        }

        void addCards(Block* block, const std::vector<ActionRecord>& records) {
            if (!block) block = addAssistantBlock({});
            for (const ActionRecord& rec : records) {
                Attachment a;
                a.kind = Attachment::Kind::Card;
                a.rec = rec;
                block->below.push_back(std::move(a));
            }
            scrollToBottom();
        }

        void onCardLink(const ActionRecord& rec) {
            Workbench& wb = app.wb();
            const std::string& link = rec.link;
            const json& state = rec.viewState;
            if (link == "undo") {
                // Undoes the card's own change, or nothing: the newest entry
                // may be a later call's, or the user's own edit. The history
                // must stand where the call left it; revisions only grow, so
                // one below that means the change is undone already.
                const History& h = wb.history();
                if (h.revision() != rec.revAfter) {
                    // A higher number is a later push, which may also follow an
                    // undo of this change: the history cannot tell which.
                    wb.logLine("Not undone: " + rec.text +
                               (h.revision() < rec.revAfter ? " (undone already)"
                                                            : " (undone already, or changes were made after it: Edit ▸ Undo steps back through them)"));
                    return;
                }
                // every entry the call pushed (an add_step with parameters makes two)
                while (h.revision() > rec.revBefore) {
                    // Opening or closing a dataset clears the history but keeps
                    // its revision, so the change can no longer be undone.
                    if (!h.canUndo()) {
                        wb.logLine("Not undone: " + rec.text + " (a dataset was opened or closed since, which clears the undo history)");
                        break;
                    }
                    const std::uint64_t at = h.revision();
                    wb.undo();
                    if (h.revision() == at) break;   // refused while a run is on (undo() logs why)
                }
            } else if (link == "view") {
                // What ToolApi saved: the step selected or viewed (numbered from
                // 1), and the view it left. The step is viewed first, since
                // view() sizes the channel list for it and the saved view then
                // sets which of them show.
                if (state.is_object()) {
                    if (state.contains("select_step") && state["select_step"].is_number_integer())
                        wb.select(state["select_step"].get<int>() - 1);
                    if (state.contains("view_step") && state["view_step"].is_number_integer()) wb.view(state["view_step"].get<int>() - 1);
                    if (state.contains("view") && state["view"].is_object()) {
                        // The card may predate a dataset load or a pipeline
                        // change: z and t are clamped to the data on screen, as
                        // set_view does, since label edits index frames by t.
                        ViewState s = ViewState::fromJson(state["view"], wb.viewState());
                        const DatasetMeta meta = wb.displayedMeta();
                        s.z = std::clamp<Index>(s.z, 0, std::max<Index>(meta.dims.z - 1, 0));
                        s.t = std::clamp<Index>(s.t, 0, std::max<Index>(meta.dims.t - 1, 0));
                        wb.setViewState(s);
                    }
                }
            } else if (link == "log") {
                app.showLog();
            }
        }

        // The last lines of the log: what a "log" link shows while hovered.
        std::string logTail() const {
            const auto& log = app.wb().log();
            std::string tail;
            const std::size_t start = log.size() > 12 ? log.size() - 12 : 0;
            for (std::size_t i = start; i < log.size(); ++i) tail += log[i] + "\n";
            tail = trimmed(tail);
            return tail.empty() ? std::string("(log is empty)") : tail;
        }

        void setBusy(bool on, const std::string& text = {}) {
            const bool was = busy;
            busy = on;
            busyBase = text.empty() ? std::string("Thinking…") : text;
            if (on) {
                busySince = Clock::now();
            } else {
                waitingForModel = false;
                reasoningChars = 0;
                if (was && refocusWhenIdle) focusRequest = true;
                refocusWhenIdle = false;
            }
            if (on) scrollToBottom();
            app.requestRedraw();
        }

        std::string busyText() const {
            std::string text = busyBase;
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - busySince).count();
            if (seconds >= 5) text += format(" · %lld s", static_cast<long long>(seconds));
            if (reasoningChars > 0)
                text += format(" · the model is reasoning before it answers (%d characters so far)", reasoningChars);
            else if (waitingForModel && seconds >= 15 && !startsWith(busyBase, "Loading "))
                text += format(" · no answer yet from %s — a large model takes a minute or two to load the first time",
                               settings.baseUrl.c_str());
            return text;
        }

        void showError(const std::string& error) {
            addAssistantBlock("⚠ " + error);
            setBusy(false);
        }

        // --- conversation ------------------------------------------------------
        json systemMessages() {
            ToolApi& api = app.tools();
            std::string prompt = api.systemPrompt();
            prompt += "\n\nCurrent workbench state (JSON):\n" + dump(api.contextSnapshot());
            json sys = {{"role", "system"}, {"content", prompt}};
            return json::array({sys});
        }

        void submit(const std::string& text) {
            const std::string t = trimmed(text);
            if (t.empty() || busy) return;
            input.clear();
            addUserBubble(t);
            history.push_back({{"role", "user"}, {"content", t}});
            rounds = 0;
            ensureModelThen([this] { step(); });
        }

        void ensureModelThen(std::function<void()> next) {
            if (!settings.model.empty()) {
                next();
                return;
            }
            setBusy(true, "Looking up models…");
            std::weak_ptr<int> token = alive;
            const int gen = generation;
            client.fetchModels(settings.baseUrl, settings.requestKey(),
                               [this, token, gen, next](std::vector<std::string> ids, std::string error) {
                                   if (token.expired() || gen != generation) return;
                                   if (ids.empty()) {
                                       showError(error.empty() ? std::string("No model configured and the server lists none. Set one "
                                                                             "in Preferences ▸ Assistant.")
                                                               : format("Cannot reach the model server at %s (%s). Configure the "
                                                                        "assistant in Preferences.",
                                                                        settings.baseUrl.c_str(), error.c_str()));
                                       return;
                                   }
                                   settings.model = ids.front();
                                   settings.save();
                                   modelText = settings.model;
                                   next();
                               });
        }

        void step() {
            // Once the window is closing no request goes out: neither the chat
            // nor the question which models Ollama holds.
            if (app.closing()) return;
            reasoningChars = 0;
            waitingForModel = true;
            setBusy(true);
            // Ollama loads a model into memory on its first request, which for
            // a large one is a minute or two of silence: ask what it holds and
            // say so right away rather than after fifteen mute seconds.
            if (settings.provider == "ollama") {
                const std::string model = settings.model;
                std::weak_ptr<int> token = alive;
                const int gen = generation;
                client.fetchLoadedModels(settings.baseUrl, [this, token, gen, model](std::vector<std::string> loaded, std::string error) {
                    if (token.expired() || gen != generation) return;
                    if (!busy || !waitingForModel || !error.empty()) return;   // no list: nothing to say
                    const bool held = std::any_of(loaded.begin(), loaded.end(), [&](const std::string& n) {
                        return n == model || n == model + ":latest" || model == n + ":latest";
                    });
                    if (held) return;
                    busyBase = format("Loading %s into memory — the first answer after a start takes a minute or two for a "
                                      "large model; later ones come in seconds",
                                      model.c_str());
                    app.requestRedraw();
                });
            }
            LlmClient::Request r;
            r.baseUrl = settings.baseUrl;
            r.model = settings.model;
            r.apiKey = settings.requestKey();
            json msgs = systemMessages();
            for (const json& m : history) msgs.push_back(m);
            r.messages = std::move(msgs);
            r.tools = app.tools().schemas();
            streamingBlock = nullptr;
            streamingLabel = false;
            streamingSource.clear();
            client.send(r);
        }

        void onDelta(const std::string& text) {
            waitingForModel = false;
            if (!streamingLabel || !streamingBlock) {
                addAssistantBlock({});
                streamingSource.clear();
            }
            streamingSource += text;
            streamingBlock->text = streamingSource;
            streamingBlock->showText = true;
            streamingBlock->dirty = true;
            scrollToBottom();
        }

        void onFinished(const json& message) {
            history.push_back(message);
            // the reply is complete: its reasoning is no news for what follows
            // ("Waiting for your confirmation…" said the model was still reasoning)
            reasoningChars = 0;
            waitingForModel = false;
            const std::string content = message.contains("content") && message["content"].is_string() ? message["content"].get<std::string>()
                                                                                                      : std::string();
            if (streamingLabel && streamingBlock) {
                streamingBlock->text = content;
                streamingBlock->showText = !content.empty();
                streamingBlock->dirty = true;
            } else if (!content.empty()) {
                addAssistantBlock(content);
            }
            pending.clear();
            if (message.contains("tool_calls") && message["tool_calls"].is_array())
                for (const json& call : message["tool_calls"]) {
                    const json fn = call.value("function", json::object());
                    PendingCall p;
                    p.id = call.value("id", std::string());
                    p.name = fn.is_object() ? fn.value("name", std::string()) : std::string();
                    p.arguments = fn.is_object() && fn.contains("arguments") && fn["arguments"].is_string() ? fn["arguments"].get<std::string>()
                                                                                                            : std::string();
                    pending.push_back(std::move(p));
                }
            if (pending.empty()) {
                setBusy(false);
                return;
            }
            if (!streamingBlock) addAssistantBlock({});
            processNextCall();
        }

        void processNextCall() {
            if (pending.empty()) {
                // let the model see the results -- unless it has been at it for too long
                if (++rounds >= kMaxRounds) {
                    showError(format("Stopped after %d rounds of tool calls without an answer: ask again to go on.", kMaxRounds));
                    return;
                }
                step();
                return;
            }
            const PendingCall call = pending.front();
            if (settings.askBeforeActing && mutatingTool(call.name)) {
                askConfirmation(call);
                return;
            }
            // Tools are called between two frames: a run pumps frames until
            // it is done, which must not happen inside the frame that
            // delivered the answer. The call stays pending until then, so
            // that a stop meanwhile answers it.
            later([this] {
                if (pending.empty()) return;
                const PendingCall next = pending.front();
                pending.pop_front();
                if (runCall(next)) processNextCall();
            });
        }

        // False when the user stopped the assistant while the call ran, or
        // closed the window.
        bool runCall(const PendingCall& call) {
            executing = true;
            executeCall(call);
            executing = false;
            // A run pumps frames, and the window may have been closed in one
            // of them: nothing more is asked of the model while the app quits.
            if (app.closing()) return false;
            if (stopAfterCall) {
                stopAfterCall = false;
                finishStop();
                return false;
            }
            return true;
        }

        void askConfirmation(const PendingCall& call) {
            setBusy(true, "Waiting for your confirmation…");
            if (!streamingBlock) addAssistantBlock({});
            Attachment a;
            a.kind = Attachment::Kind::Confirm;
            a.call = call;
            streamingBlock->below.push_back(std::move(a));
            scrollToBottom();
        }

        // Apply / Skip on a confirmation card: answered once. Apply can start
        // a run, which pumps frames until it is done; the card is gone from
        // the first of them, so a second click can neither pop the next
        // pending call without running or answering it, nor answer this one
        // twice, nor send a chat request in the middle of the run.
        void answer(Attachment& card, bool doIt) {
            if (card.answered) return;
            card.answered = true;
            const PendingCall call = card.call;
            later([this, call, doIt] {
                if (!pending.empty()) pending.pop_front();
                if (doIt) {
                    if (!runCall(call)) return;
                } else {
                    history.push_back({{"role", "tool"}, {"tool_call_id", call.id}, {"content", "{\"error\":\"the user declined this action\"}"}});
                }
                setBusy(true);
                processNextCall();
            });
        }

        void executeCall(const PendingCall& call) {
            setBusy(true, format("Running %s…", call.name.c_str()));
            // Arguments that are not a JSON object -- typically a reply cut
            // off at the token limit -- go back to the model as an error.
            // They used to run as {}: a `run` cut short ran every step, a
            // set_step_param with its value missing reset to defaults.
            const json args = trimmed(call.arguments).empty() ? json::object() : json::parse(call.arguments, nullptr, false);
            json result;
            ToolApi& api = app.tools();
            if (args.is_discarded() || !args.is_object()) {
                result = {{"error", "the arguments of this call are not a valid JSON object (was the reply cut off?); nothing was done"}};
            } else if (mutatingTool(call.name) && app.bridge().taskRunning()) {
                // A dataset load that finishes installs its dataset with a
                // fresh Load step and clears the history: a change made while
                // it runs would be lost, although it was reported as done. The
                // window refuses edits during any task (an export too), so the
                // error names the task that is running.
                result = {{"error", app.bridge().taskLabel() + " is in progress: wait for it to finish before changing anything"}};
            } else {
                try {
                    result = api.call(call.name, args);
                } catch (const std::exception& e) {
                    result = {{"error", e.what()}};
                }
            }
            const std::vector<ActionRecord> actions = api.takeActions();
            if (!actions.empty()) addCards(streamingBlock, actions);
            bool cut = false;
            std::string content = leftChars(dump(result), kMaxToolResult, &cut);
            if (cut) content += "…(truncated)";
            history.push_back({{"role", "tool"}, {"tool_call_id", call.id}, {"name", call.name}, {"content", content}});
        }

        // The stop button: the request in flight is dropped, the calls not
        // yet made are answered as not done (the conversation must hold an
        // answer to every call before the next question), a run a tool
        // started is cancelled.
        void stop() {
            if (!busy) return;
            ++generation;
            client.abort();
            for (const PendingCall& call : pending)
                history.push_back(
                    {{"role", "tool"}, {"tool_call_id", call.id}, {"content", "{\"error\":\"the user stopped the assistant before this ran\"}"}});
            pending.clear();
            for (auto& b : blocks)
                for (Attachment& a : b->below)
                    if (a.kind == Attachment::Kind::Confirm) a.answered = true;
            if (executing) {
                // the call finishes first (a run winds down); the loop ends there
                stopAfterCall = true;
                if (app.bridge().running()) app.bridge().cancelRun();
                setBusy(true, "Stopping…");
                return;
            }
            finishStop();
        }

        void finishStop() {
            addAssistantBlock("Stopped.");
            setBusy(false);
        }

        std::string contextLine() const {
            const Workbench& wb = app.wb();
            const int sel = wb.selectedIndex();
            // With no step there is nothing to see: the line said "sees step
            // 01" over an empty pipeline.
            const int steps = wb.pipeline().size();
            if (sel < 0 || sel >= steps) return steps == 0 ? "no steps yet" : "sees the ops stack";
            return "sees step " + Step::number(sel) + ", diagnostics, ops stack";
        }

        bool unlisted() const {
            return !listedModels.empty() && std::find(listedModels.begin(), listedModels.end(), settings.model) == listedModels.end();
        }

        // A pick or a typed name in the footer is the setting, saved at once.
        void commitModel(const std::string& text) {
            const std::string m = LlmClient::resolveModel(text, listedModels);
            if (m.empty()) return;
            if (m != trimmed(text)) modelText = m;
            if (m == settings.model) return;
            settings.model = m;
            settings.save();
            std::string line = "Assistant model: " + m;
            if (unlisted())
                line += " (not among the " + std::to_string(listedModels.size()) + " the server lists; the answer will say if it is unknown)";
            app.wb().logLine(line);
        }

        // The server's model list into the footer's dropdown, the current
        // choice kept (and listed even when the server does not know it).
        void refreshModelBox() {
            const std::string keep = settings.model;
            listedModels.clear();
            modelItems.clear();
            if (!keep.empty()) modelItems.push_back(keep);
            modelText = keep;
            const int lookup = ++modelLookup;
            std::weak_ptr<int> token = alive;
            client.fetchModels(settings.baseUrl, settings.requestKey(), [this, token, lookup](std::vector<std::string> ids, std::string error) {
                if (token.expired() || lookup != modelLookup) return;   // a newer list was asked for
                if (ids.empty()) {
                    modelTip = format("Cannot list the models at %s (%s): type a name", settings.baseUrl.c_str(), error.c_str());
                    return;
                }
                std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) { return toLower(a) < toLower(b); });
                listedModels = ids;
                // The choice as it is now, not as it was when the list was
                // asked for: a name typed or picked meanwhile stands. One that
                // is only a prefix of a name the server has (typed and left,
                // say) becomes that name.
                const std::string current = settings.model;
                const std::string chosen = current.empty() ? ids.front() : LlmClient::resolveModel(current, ids);
                if (std::find(ids.begin(), ids.end(), chosen) == ids.end()) ids.insert(ids.begin(), chosen);
                modelItems = ids;
                if (!modelEditing) modelText = chosen;
                modelTip = format("%d model(s) at %s: pick one, or type a name", static_cast<int>(ids.size()), settings.baseUrl.c_str());
                if (chosen != current) commitModel(chosen);
            });
        }

        // --- drawing -------------------------------------------------------------------

        void layoutBlock(Block& b, float width, ImU32 color) {
            if (!b.dirty && b.layout.width == width && b.layout.scale == theme::scale()) return;
            const md::Document doc = b.user ? md::plain(b.text) : md::parse(b.text);
            b.layout = md::layout(doc, width, theme::kBodyPx, color);
            b.dirty = false;
        }

        // Right-click on a message: copy it, or the whole conversation.
        void offerCopy(ImVec2 min, ImVec2 max, const std::string& text) {
            if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(min, max) && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                copyText = text;
                ImGui::OpenPopup("##copy");
            }
        }

        std::string conversationText() const {
            std::string out;
            for (const auto& b : blocks) {
                std::string part;
                if (b->showText && !b->text.empty()) part = (b->user ? "You: " : "Assistant: ") + b->text;
                for (const Attachment& a : b->below)
                    if (a.kind == Attachment::Kind::Card) part += (part.empty() ? "" : "\n") + std::string("  · ") + a.rec.text;
                if (part.empty()) continue;
                if (!out.empty()) out += "\n\n";
                out += part;
            }
            return out;
        }

        void drawCopyMenu() {
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 4));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
            if (ImGui::BeginPopup("##copy")) {
                auto item = [](const char* label) {
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const float h = theme::snap(px(26));
                    const bool clicked = ImGui::Selectable((std::string("##") + label).c_str(), false, ImGuiSelectableFlags_None,
                                                           ImVec2(std::max(px(160), ImGui::GetContentRegionAvail().x), h));
                    widgets::drawTextIn(ImGui::GetWindowDrawList(), ImVec2(p.x + px(12), p.y), ImVec2(p.x + px(300), p.y + h), label, 12,
                                        theme::kText, theme::Weight::Regular, 0.0f, 0.5f);
                    return clicked;
                };
                if (item("Copy")) ImGui::SetClipboardText(copyText.c_str());
                if (item("Copy conversation")) ImGui::SetClipboardText(conversationText().c_str());
                ImGui::EndPopup();
            }
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }

        // Draws a laid-out text at `origin`; a click on a link opens it.
        // The model wrote the target, so it is shown while the link is
        // hovered, and only a web address is handed to the shell (which
        // would start a program or a protocol handler just as readily).
        void drawText(const md::Layout& layout, ImVec2 origin) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const int link = md::draw(dl, origin, layout);
            if (link >= 0 && link < static_cast<int>(layout.links.size())) {
                const std::string& href = layout.links[static_cast<std::size_t>(link)];
                // an item under the text, for the tooltip to hang on
                place(origin.x, origin.y);
                ImGui::Dummy(layout.size);
                widgets::tooltip(href);
                if (md::isWebUrl(href)) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) platform::openUrl(href);
                }
            }
        }

        // One action card: glyph | text | link. Returns its height.
        float drawCard(const Attachment& a, int index, float x, float y, float w) {
            const ActionRecord& rec = a.rec;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float padX = px(10), padY = px(7);
            const float lh = lineHeight(12);
            const float h = theme::snap(padY * 2 + std::max(px(14), lh));
            Icon icon = Icon::Info;
            bool accent = false;
            switch (rec.kind) {
                case ActionRecord::Kind::Param: icon = Icon::Pencil; break;
                case ActionRecord::Kind::Run: icon = Icon::Play; break;
                case ActionRecord::Kind::View:
                    icon = Icon::Eye;
                    accent = true;
                    break;
                case ActionRecord::Kind::Edit: icon = Icon::Pencil; break;
                case ActionRecord::Kind::Info: break;
            }
            const float midY = y + h * 0.5f;
            drawIcon(dl, ImVec2(x + padX, midY - px(7)), ImVec2(x + padX + px(14), midY + px(7)), icon, accent ? theme::kAccent : theme::kText,
                     px(1.5f));
            float right = x + w - padX;
            ImGui::PushID(index);
            if (!rec.link.empty()) {
                const ImVec2 ls = theme::textSize(rec.link, 12);
                const ImVec2 lmin(right - ls.x, midY - ls.y * 0.5f);
                place(lmin.x, lmin.y);
                ImGui::InvisibleButton("##link", ls);
                const bool hot = ImGui::IsItemHovered();
                if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    // after the transcript was walked: undo and view change what it shows
                    std::weak_ptr<int> token = alive;
                    app.defer([this, token, rec] {
                        if (!token.expired()) onCardLink(rec);
                    });
                }
                if (rec.link == "log") widgets::tooltip(logTail());
                widgets::drawText(dl, lmin, rec.link, 12, hot ? theme::kAccent700 : theme::kAccentText);
                if (hot) dl->AddRectFilled(ImVec2(lmin.x, lmin.y + ls.y), ImVec2(lmin.x + ls.x, lmin.y + ls.y + theme::crispPen(1)), theme::kAccent700);
                right = lmin.x - px(8);
            }
            const float textX = x + padX + px(18) + px(8);
            const float room = std::max(0.0f, right - textX);
            const std::string shown = widgets::elideText(rec.text, room, 12);
            widgets::drawText(dl, ImVec2(textX, midY - lh * 0.5f), shown, 12, theme::kText);
            place(textX, y);
            ImGui::InvisibleButton("##text", ImVec2(std::max(1.0f, room), h));
            widgets::tooltip(rec.text);
            ImGui::PopID();
            widgets::crispRect(dl, ImVec2(x, y), ImVec2(x + w, y + h), theme::kDivider, theme::kBorder);
            offerCopy(ImVec2(x, y), ImVec2(x + w, y + h), rec.text);
            return h;
        }

        // "Apply set_param {…}?" with Apply / Skip. Returns its height.
        float drawConfirm(Attachment& a, int index, float x, float y, float w) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float padX = px(10), padY = px(7), gap = px(8);
            const ImVec2 apply = chipSize("Apply"), skip = chipSize("Skip");
            const float textW = std::max(px(40), w - 2 * padX - apply.x - skip.x - 2 * gap);
            const std::string text = "Apply " + a.call.name + " " + leftChars(a.call.arguments, 80) + "?";
            float textH = 0.0f;
            {
                const theme::FontScope f(12);
                textH = ImGui::CalcTextSize(text.c_str(), nullptr, false, textW).y;
            }
            const float h = theme::snap(2 * padY + std::max(textH, apply.y));
            place(x + padX, y + (h - textH) * 0.5f);
            widgets::textWrapped(text, 12, theme::kText, theme::Weight::Regular, textW);
            ImGui::PushID(index);
            const float cy = y + (h - apply.y) * 0.5f;
            place(x + w - padX - skip.x - gap - apply.x, cy);
            if (widgets::chipButton("Apply##apply")) answer(a, true);
            place(x + w - padX - skip.x, cy);
            if (widgets::chipButton("Skip##skip")) answer(a, false);
            ImGui::PopID();
            widgets::crispRect(dl, ImVec2(x, y), ImVec2(x + w, y + h), theme::kAccent, theme::kBorder);
            return h;
        }

        void drawTranscript(float width, float height) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBg);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            if (ImGui::BeginChild("##transcript", ImVec2(width, height), ImGuiChildFlags_None, ImGuiWindowFlags_None)) {
                const ImVec2 o = ImGui::GetCursorScreenPos();
                // the viewport (without the scroll bar); the labels wrap at a share of it
                const float viewW = std::max(px(60), ImGui::GetContentRegionAvail().x);
                const float margin = px(14), spacing = px(14);
                const float left = o.x + margin;
                const float contentW = std::max(px(40), viewW - 2 * margin);
                float y = o.y + margin;
                bool first = true;
                int cardIndex = 0;
                for (auto& bp : blocks) {
                    Block& b = *bp;
                    const bool anything =
                        (b.showText && !b.text.empty()) ||
                        std::any_of(b.below.begin(), b.below.end(), [](const Attachment& a) { return a.kind == Attachment::Kind::Card || !a.answered; });
                    if (!anything) continue;
                    if (!first) y += spacing;
                    first = false;
                    if (b.user) {
                        // ink bubble, paper text, right-aligned, at most 88 % of the width
                        const float maxW = std::min(contentW, std::floor(viewW * 0.88f));
                        const float padX = px(12), padY = px(8);
                        layoutBlock(b, std::max(px(20), maxW - 2 * padX), theme::kBg);
                        const float bw = theme::snap(std::min(maxW, b.layout.size.x + 2 * padX));
                        const float bh = theme::snap(b.layout.size.y + 2 * padY);
                        const ImVec2 min(left + contentW - bw, y), max(left + contentW, y + bh);
                        ImGui::GetWindowDrawList()->AddRectFilled(min, max, theme::kText);
                        drawText(b.layout, ImVec2(min.x + padX, min.y + padY));
                        offerCopy(min, max, b.text);
                        y += bh;
                        continue;
                    }
                    const float blockW = std::min(contentW, std::floor(viewW * 0.94f));
                    bool firstPart = true;
                    if (b.showText && !b.text.empty()) {
                        layoutBlock(b, blockW, theme::kText);
                        drawText(b.layout, ImVec2(left, y));
                        offerCopy(ImVec2(left, y), ImVec2(left + blockW, y + b.layout.size.y), b.text);
                        y += b.layout.size.y;
                        firstPart = false;
                    }
                    for (Attachment& a : b.below) {
                        if (a.kind == Attachment::Kind::Confirm && a.answered) continue;
                        if (!firstPart) y += px(8);
                        firstPart = false;
                        y += a.kind == Attachment::Kind::Card ? drawCard(a, cardIndex, left, y, blockW) : drawConfirm(a, cardIndex, left, y, blockW);
                        ++cardIndex;
                    }
                }
                // busy: accent square + what it is waiting for
                if (busy) {
                    if (!first) y += spacing;
                    const float lh = lineHeight(12);
                    const float sq = theme::snap(px(8));
                    const float textW = std::max(px(40), contentW - sq - px(8));
                    const std::string text = busyText();
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(left, theme::snap(y + (lh - sq) * 0.5f)),
                                                              ImVec2(left + sq, theme::snap(y + (lh - sq) * 0.5f) + sq), theme::kAccent);
                    place(left + sq + px(8), y);
                    // it grows into a sentence while the wait goes on: wrapped
                    widgets::textWrapped(text, 12, theme::kNeutral600, theme::Weight::Regular, textW);
                    y = ImGui::GetItemRectMax().y;
                }
                reach(o.x, y + margin);
                drawCopyMenu();
                if (scrollDown > 0) {
                    ImGui::SetScrollY(ImGui::GetScrollMaxY() + px(4000));
                    --scrollDown;
                    app.requestRedraw();
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        }

        // The question: several lines, Enter sends, Shift+Enter starts a new line.
        bool questionField(float w, float h, bool enabled, float padY) {
            ImGui::PushFont(theme::font(), theme::kBodyPx);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(8), padY));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, theme::kBg);
            ImGui::PushStyleColor(ImGuiCol_Border, enabled ? theme::kDivider : theme::kNeutral300);
            ImGui::BeginDisabled(!enabled);
            if (enabled && focusRequest) {
                ImGui::SetKeyboardFocusHere();
                focusRequest = false;
            }
            const ImGuiInputTextFlags flags =
                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine | ImGuiInputTextFlags_WordWrap;
            const bool entered = ImGui::InputTextMultiline("##question", &input, ImVec2(w, h), flags);
            const bool active = ImGui::IsItemActive();
            const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (active) widgets::crispRect(dl, min, max, theme::kAccent, 2.0f);
            if (input.empty() && !active)
                widgets::drawText(dl, ImVec2(min.x + px(8) + theme::crispPen(theme::kBorder), min.y + padY), "Ask, or tell it what to do…",
                                  theme::kBodyPx, theme::kNeutral500);
            if (ImGui::IsItemHovered() && enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
            ImGui::EndDisabled();
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar(2);
            ImGui::PopFont();
            return entered;
        }

        // The 34 px square next to the question: send (↵), or stop while busy.
        bool sendButton(float side, bool stopping) {
            const ImVec2 min = ImGui::GetCursorScreenPos();
            const bool pressed = ImGui::InvisibleButton("##send", ImVec2(side, side));
            const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
            if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            widgets::tooltip(stopping ? std::string("Stop") : widgets::withShortcut("Send", shortcutText(keys::send)));
            const ImVec2 max(min.x + side, min.y + side);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(min, max, held ? theme::kAccent700 : (hovered ? theme::kAccent600 : theme::kAccent));
            const ImVec2 c((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f);
            if (stopping) {
                const float s = theme::snap(px(10));
                dl->AddRectFilled(ImVec2(theme::snap(c.x - s * 0.5f), theme::snap(c.y - s * 0.5f)),
                                  ImVec2(theme::snap(c.x - s * 0.5f) + s, theme::snap(c.y - s * 0.5f) + s), theme::kBg);
            } else {
                drawIcon(dl, c, px(14), Icon::Enter, theme::kBg, px(1.5f));
            }
            return pressed;
        }

        void draw() {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kBg);
            const float rule = theme::crispPen(theme::kRule);
            const float margin = px(14);
            const float innerW = std::max(px(40), avail.x - 2 * margin);

            // header: ✦ Assistant · context · ✕
            const float headerH = theme::snap(px(40));
            {
                const float cy = origin.y + headerH * 0.5f;
                float x = origin.x + margin;
                drawIcon(dl, ImVec2(x, cy - px(7)), ImVec2(x + px(14), cy + px(7)), Icon::Sparkle, theme::kAccent, px(1.5f));
                x += px(14) + px(10);
                const ImVec2 ts = theme::textSize("Assistant", 13, theme::Weight::ExtraBold);
                widgets::drawText(dl, ImVec2(x, cy - ts.y * 0.5f), "Assistant", 13, theme::kText, theme::Weight::ExtraBold);
                x += ts.x + px(10);
                const float closeSide = px(18);
                const float closeX = origin.x + avail.x - margin - closeSide;
                const float room = closeX - px(10) - x;
                if (room > px(12)) {
                    place(x, cy - lineHeight(theme::kSmallPx) * 0.5f);
                    widgets::elided(contextLine(), room, theme::kSmallPx, theme::kNeutral600);
                }
                place(closeX, cy - closeSide * 0.5f);
                widgets::GlyphOpts close;
                close.borderless = true;
                close.iconPx = 11;
                close.tooltip = "Close the assistant";
                if (widgets::glyphButton("##closeAssistant", Icon::Close, 18, close)) app.setAssistantVisible(false);
                dl->AddRectFilled(ImVec2(origin.x, origin.y + headerH), ImVec2(origin.x + avail.x, origin.y + headerH + rule), theme::kDivider);
            }

            // footer metrics: it keeps its place at the bottom of the dock
            static const std::vector<std::string> suggestions{"Explain the Wiener parameter", "Why is step 06 skipped?",
                                                              "Max-project over Z and show me", "Flag low-confidence labels"};
            std::vector<ImVec2> chip;
            for (const std::string& s : suggestions) chip.push_back(chipSize(s));
            const float gap6 = px(6);
            const float col0 = std::max(chip[0].x, chip[2].x), col1 = std::max(chip[1].x, chip[3].x);
            // two to a row, as in the design, when the dock is wide enough; one to a row when not
            const bool twoColumns = col0 + gap6 + col1 <= innerW;
            const float chipH = chip[0].y;
            const int chipRows = twoColumns ? 2 : 4;
            const float chipsH = chipRows * chipH + (chipRows - 1) * gap6;

            const float side = theme::snap(px(34));
            const float fieldW = std::max(px(40), innerW - side - gap6);
            const float lh = fontPx(theme::kBodyPx);
            const float padY = std::max(px(2), std::floor((side - lh) * 0.5f));
            float textH = lh;
            {
                const float wrapW = std::max(px(10), fieldW - 2 * px(8) - ImGui::GetStyle().ScrollbarSize);
                ImFont* f = theme::font() ? theme::font() : ImGui::GetFont();
                textH = std::max(lh, f->CalcTextSizeA(lh, FLT_MAX, wrapW, input.c_str(), input.c_str() + input.size()).y);
                if (!input.empty() && input.back() == '\n') textH += lh;
            }
            const float inputH = theme::snap(std::clamp(textH + 2 * padY, side, std::max(side, px(120))));
            const float modelH = theme::snap(px(30));
            const float noteH = lineHeight(theme::kSmallPx);
            const float footerH = rule + px(12) + chipsH + px(10) + inputH + px(10) + modelH + px(10) + noteH + px(12);

            // transcript
            const float top = origin.y + headerH + rule;
            const float transcriptH = std::max(px(40), avail.y - headerH - rule - footerH);
            place(origin.x, top);
            drawTranscript(avail.x, transcriptH);

            // footer
            const float fy = top + transcriptH;
            dl->AddRectFilled(ImVec2(origin.x, theme::snap(fy)), ImVec2(origin.x + avail.x, theme::snap(fy) + rule), theme::kDivider);
            float y = fy + rule + px(12);
            const float x0 = origin.x + margin;
            for (std::size_t i = 0; i < suggestions.size(); ++i) {
                const float cx = twoColumns ? x0 + (i % 2 == 0 ? 0.0f : col0 + gap6) : x0;
                const float cy = twoColumns ? y + static_cast<float>(i / 2) * (chipH + gap6) : y + static_cast<float>(i) * (chipH + gap6);
                place(cx, cy);
                ImGui::PushID(static_cast<int>(i));
                widgets::ButtonOpts o;
                o.kind = widgets::ButtonKind::Chip;
                if (chip[i].x > innerW) o.width = dp(innerW);
                if (widgets::button((suggestions[i] + "##chip").c_str(), o)) {
                    input = suggestions[i];
                    focusRequest = true;
                }
                ImGui::PopID();
            }
            y += chipsH + px(10);

            place(x0, y);
            const bool entered = questionField(fieldW, inputH, !busy, padY);
            place(x0 + innerW - side, y);
            const bool clicked = sendButton(side, busy);
            if ((entered || (clicked && !busy)) && !busy) {
                const std::string text = input;
                // Enter leaves the field: it takes the keyboard back once the answer is in
                if (entered) {
                    if (trimmed(text).empty()) focusRequest = true;
                    else refocusWhenIdle = true;
                }
                submit(text);
            } else if (clicked && busy) {
                stop();
            }
            y += inputH + px(10);

            // the model that answers, where the question is asked
            {
                const float cy = y + modelH * 0.5f;
                const ImVec2 ls = theme::textSize("Model", theme::kSmallPx);
                widgets::drawText(dl, ImVec2(x0, cy - ls.y * 0.5f), "Model", theme::kSmallPx, theme::kNeutral600);
                const std::string ask = settings.askBeforeActing ? "Ask before acting ✓" : "Ask before acting";
                const ImVec2 as = theme::textSize(ask, theme::kSmallPx);
                const float comboX = x0 + ls.x + gap6;
                const float comboW = std::max(px(60), x0 + innerW - as.x - px(10) - comboX);
                place(comboX, y);
                widgets::FieldOpts fo;
                fo.width = dp(comboW);
                fo.height = dp(modelH);
                const bool changed = widgets::editableCombo("##model", &modelText, modelItems, fo);
                // the last item is the dropdown's text field
                const bool editingNow = ImGui::IsItemActive();
                if (ImGui::IsItemDeactivatedAfterEdit()) commitModel(modelText);   // typed, and left (Enter, a click away)
                else if (changed && !editingNow) commitModel(modelText);           // picked from the list
                modelEditing = editingNow;
                // outlined in the accent while its name is one the server does
                // not list (a typed name, or a server that changed)
                const bool off = unlisted();
                widgets::tooltip(off ? format("%s is not a model %s lists: pick one from the list, or the answer will say if it is unknown",
                                              settings.model.c_str(), settings.baseUrl.c_str())
                                     : modelTip);
                if (off) widgets::crispRect(dl, ImVec2(comboX, y), ImVec2(comboX + theme::snap(comboW), y + modelH), theme::kAccent, theme::kBorder);
                place(x0 + innerW - as.x, cy - as.y * 0.5f);
                if (widgets::linkButton((ask + "##ask").c_str())) {
                    settings.askBeforeActing = !settings.askBeforeActing;
                    settings.save();
                }
            }
            y += modelH + px(10);
            widgets::drawText(dl, ImVec2(x0, y), "Changes are applied as undoable steps", theme::kSmallPx, theme::kNeutral600);
            reach(origin.x, origin.y + avail.y);
            ImGui::PopStyleVar();
        }
    };

    AssistantPanel::AssistantPanel(App& app) : impl_(std::make_unique<Impl>(app)) {}
    AssistantPanel::~AssistantPanel() = default;

    void AssistantPanel::draw() { impl_->draw(); }

    void AssistantPanel::setSettings(const AssistantSettings& s) {
        impl_->settings = s;
        impl_->settings.save();
        impl_->refreshModelBox();
    }

    AssistantSettings AssistantPanel::settings() const { return impl_->settings; }

    void AssistantPanel::focusInput() {
        impl_->focusRequest = true;
        impl_->app.requestRedraw();
    }

    void AssistantPanel::ask(const std::string& text) { impl_->submit(text); }

    bool AssistantPanel::busy() const { return impl_->busy; }

} // namespace sirius::app::gui
