#ifndef SIRIUS_APP_BYTE_BUDGET_LRU_HPP
#define SIRIUS_APP_BYTE_BUDGET_LRU_HPP

// Which entries of a cache to drop so that it stays within a byte budget,
// least recently used first. It holds keys and sizes only; the cache keeps the
// values and erases what evict() names.
//
// For the viewer's per-(channel, time point) projections: they were kept for
// every time point with no limit, which is 16 MB each at 2048 x 2048, so a
// two-channel, 500-frame movie grew to 32 GB once play had looped.

#include <cstddef>
#include <list>
#include <map>
#include <utility>
#include <vector>

namespace sirius::app {

    template <typename Key> class ByteBudgetLru {
    public:
        explicit ByteBudgetLru(std::size_t budgetBytes) : budget_(budgetBytes) {}

        std::size_t budget() const noexcept { return budget_; }
        std::size_t bytes() const noexcept { return bytes_; }
        std::size_t size() const noexcept { return index_.size(); }

        // `key` was written (or written again) with `bytes`; it becomes the
        // most recently used. Returns the keys to erase, oldest first. The key
        // just written is never among them, even when it alone is over budget:
        // the frame on screen has to stay.
        std::vector<Key> put(const Key& key, std::size_t bytes) {
            erase(key);
            order_.push_back({key, bytes});
            index_[key] = std::prev(order_.end());
            bytes_ += bytes;
            std::vector<Key> evicted;
            while (bytes_ > budget_ && order_.size() > 1) {
                const auto oldest = order_.begin();
                bytes_ -= oldest->second;
                evicted.push_back(oldest->first);
                index_.erase(oldest->first);
                order_.erase(oldest);
            }
            return evicted;
        }

        // `key` was read: it becomes the most recently used.
        void touch(const Key& key) {
            const auto it = index_.find(key);
            if (it == index_.end()) return;
            order_.splice(order_.end(), order_, it->second);
        }

        void erase(const Key& key) {
            const auto it = index_.find(key);
            if (it == index_.end()) return;
            bytes_ -= it->second->second;
            order_.erase(it->second);
            index_.erase(it);
        }

        void clear() {
            order_.clear();
            index_.clear();
            bytes_ = 0;
        }

    private:
        using Order = std::list<std::pair<Key, std::size_t>>;
        std::size_t budget_;
        std::size_t bytes_ = 0;
        Order order_;   // least recently used first
        std::map<Key, typename Order::iterator> index_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_BYTE_BUDGET_LRU_HPP
