// ByteBudgetLru: which cache entries go when a byte budget is exceeded
// (app/core/byte_budget_lru.hpp; the viewer's per-time-point projections).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/byte_budget_lru.hpp"

using sirius::app::ByteBudgetLru;

TEST_CASE("ByteBudgetLru drops the least recently used entries past its budget", "[app][lru]") {
    using Key = std::pair<int, int>;   // (channel, time point)
    ByteBudgetLru<Key> lru(100);
    CHECK(lru.put({0, 0}, 40).empty());
    CHECK(lru.put({0, 1}, 40).empty());
    CHECK(lru.bytes() == 80);

    // over budget: the oldest goes, the one just written never does
    CHECK(lru.put({0, 2}, 40) == std::vector<Key>{{0, 0}});
    CHECK(lru.bytes() == 80);
    CHECK(lru.size() == 2);

    // a read makes an entry the most recent, so play that loops keeps what it shows
    lru.touch({0, 1});
    CHECK(lru.put({0, 3}, 40) == std::vector<Key>{{0, 2}});

    // written again with another size: counted once
    CHECK(lru.put({0, 3}, 10).empty());
    CHECK(lru.bytes() == 50);

    // one entry larger than the whole budget stays (it is the frame on screen)
    // and takes everything else with it
    const std::vector<Key> evicted = lru.put({1, 0}, 500);
    CHECK(evicted == std::vector<Key>{{0, 1}, {0, 3}});
    CHECK(lru.size() == 1);
    CHECK(lru.bytes() == 500);

    lru.erase({1, 0});
    CHECK(lru.bytes() == 0);
    lru.touch({9, 9});   // unknown keys are ignored
    lru.erase({9, 9});
    CHECK(lru.put({2, 0}, 60).empty());
    lru.clear();
    CHECK(lru.size() == 0);
    CHECK(lru.bytes() == 0);
}

TEST_CASE("A movie of projections stays within the cache budget", "[app][lru]") {
    // two channels, 500 frames of 2048 x 2048 float32: 32 GB with no limit
    const std::size_t mip = std::size_t{2048} * 2048 * sizeof(float);
    ByteBudgetLru<std::pair<int, int>> lru(std::size_t{1} << 30);
    std::size_t peak = 0, evictions = 0;
    for (int loop = 0; loop < 2; ++loop)
        for (int t = 0; t < 500; ++t)
            for (int c = 0; c < 2; ++c) {
                evictions += lru.put({c, t}, mip).size();
                peak = std::max(peak, lru.bytes());
            }
    CHECK(peak <= (std::size_t{1} << 30));
    CHECK(lru.size() == 64);
    CHECK(evictions == 2 * 500 * 2 - 64);
}
