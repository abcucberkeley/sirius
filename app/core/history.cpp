#include "core/history.hpp"

#include <utility>

namespace sirius::app {

    void History::push(Command c) {
        redo_.clear();
        const std::uint64_t revision = ++lastRevision_;
        if (mergesWith(c.mergeKey)) {
            undo_.back() = {std::move(c), revision};   // the newest command spans the whole group
            return;
        }
        mergeOpen_ = !c.mergeKey.empty();
        undo_.push_back({std::move(c), revision});
        if (undo_.size() > limit_) {
            base_ = undo_.front().revision;   // what undoing everything that is left comes back to
            undo_.erase(undo_.begin());
        }
    }

    bool History::mergesWith(const std::string& key) const noexcept {
        return mergeOpen_ && !key.empty() && !undo_.empty() && undo_.back().command.mergeKey == key;
    }

    std::string History::undoLabel() const { return undo_.empty() ? std::string() : undo_.back().command.label; }
    std::string History::redoLabel() const { return redo_.empty() ? std::string() : redo_.back().command.label; }

    void History::undo() {
        mergeOpen_ = false;   // an undo ends the group, even when it uncovers it again
        if (undo_.empty()) return;
        Entry e = std::move(undo_.back());
        undo_.pop_back();
        if (e.command.undo) e.command.undo();
        e.command.mergeKey.clear();   // a redone command never merges
        redo_.push_back(std::move(e));
    }

    void History::redo() {
        mergeOpen_ = false;
        if (redo_.empty()) return;
        Entry e = std::move(redo_.back());
        redo_.pop_back();
        if (e.command.redo) e.command.redo();
        undo_.push_back(std::move(e));
    }

    void History::clear() {
        mergeOpen_ = false;
        base_ = revision();
        undo_.clear();
        redo_.clear();
    }

} // namespace sirius::app
