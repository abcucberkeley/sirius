// The raw SIM storage layout (core/dataset.hpp: SimLayout, SimStorage,
// SimFrames): the text form and its parser, the binding of a layout to a
// dataset's dims with its arithmetic errors, and the frame arithmetic --
// which for the z-packed shorthand has to give exactly the section indices
// the reconstructor and the SIM step have always used.
//
// Three real shapes drive it: cudasirecon's raw.tif (135 = 3 angles x 5
// phases x 9 z packed on z), mcSIM's synthetic_microtubules.tiff (c3 z3
// y2048 x2048: the angle on the channel axis, the phase on z) and OpenSIM's
// sim01z4.tif (a 1536 x 1536 plane that is a 3 x 3 montage of 512 x 512
// frames). The big ones are dims only: nothing here reads a file.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/dataset.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::ContainsSubstring;

namespace {
    // The section index the SIM step and the library's reorderFrames used
    // for a z-packed stack before layouts existed (app/core/ops/sim.cpp
    // sectionIndex, src/sim_cpu.cpp reorderFrames).
    Index oldFormula(int ndirs, int nphases, Index nz, bool fastSi, Index d, Index z, Index phase) {
        if (fastSi) return (z * ndirs + d) * nphases + phase;
        return (d * nz + z) * nphases + phase;
    }
} // namespace

TEST_CASE("SIM layout: the shorthand expands into the z-packed text and back", "[app][sim_layout]") {
    const SimLayout plain = SimLayout::shorthand(3, 5);
    CHECK(plain.present);
    CHECK(plain.isShorthand());
    CHECK(plain.text() == "z=[angle 3, z, phase 5]");
    CHECK(SimLayout::shorthand(3, 5, true).text() == "z=[z, angle 3, phase 5]");
    CHECK(plain.sectionsPerPlane() == 15);

    const SimLayout general = SimLayout::fromText("z=[angle 3, z, phase 5]");
    CHECK(general.present);
    CHECK_FALSE(general.isShorthand());
    CHECK(general.storage == "z=[angle 3, z, phase 5]");
    CHECK(general.ndirs == 3);
    CHECK(general.nphases == 5);
    CHECK_FALSE(general.fastSi);
    const SimLayout fast = SimLayout::fromText("z=[z, angle 3, phase 5]");
    CHECK(fast.fastSi);
    CHECK(fast.ndirs == 3);
    CHECK(fast.nphases == 5);

    SECTION("the text is canonical whatever the spelling") {
        CHECK(SimLayout::fromText("Z = [ DIRS 3 , z , Phases 5 ]").storage == "z=[angle 3, z, phase 5]");
        CHECK(SimLayout::fromText("z=[angle3,z,phase5];").storage == "z=[angle 3, z, phase 5]");
        CHECK(SimLayout::fromText("z = phase 3 ; c = angle 3").storage == "c=angle 3; z=phase 3");
        CHECK(SimLayout::fromText("yx=[angle 3, phase 3]").storage == "yx=3x3[angle 3, phase 3]");
        CHECK(SimLayout::fromText("yx = 3 x 5 [ angle 3 , phase 5 ]").storage == "yx=3x5[angle 3, phase 5]");
    }
    SECTION("an axis that is itself is left out of the text, written down or not") {
        // the canonical text has to be the one a user writes, so that a layout
        // bound to a dataset -- where bindSimLayout gives every axis a factor,
        // c and t included -- reads back as what was written. Without that the
        // text of a bound raw.tif layout grew a "c=c 1; t=t 1;" prefix.
        CHECK(SimLayout::fromText("c=c 2; z=[angle 3, z, phase 5]").storage == "z=[angle 3, z, phase 5]");
        CHECK(SimLayout::fromText("t=t 4; c=angle 3; z=phase 3").storage == "c=angle 3; z=phase 3");
        // and the fast-SI mirror still sees a z-packed layout through it
        CHECK(SimLayout::fromText("c=c 2; z=[z, angle 3, phase 5]").fastSi);
        // a factor beside its own kind is not the identity and stays
        CHECK(SimLayout::fromText("c=[c 2, angle 3]; z=phase 5").storage == "c=[c 2, angle 3]; z=phase 5");
        // the three real shapes: the bound layout's text is a layout again --
        // the remainders filled in, nothing else added -- and it binds to the
        // same frames, so the text can be written down and reopened
        const std::pair<const char*, Dims5> real[3] = {{"z=[angle 3, z, phase 5]", Dims5{1, 1, 135, 64, 64}},
                                                       {"c=angle 3; z=phase 3", Dims5{3, 1, 3, 64, 64}},
                                                       {"yx=3x3[angle 3, phase 3]", Dims5{1, 1, 1, 192, 192}}};
        for (const auto& [text, dims] : real) {
            INFO(text);
            const SimLayout layout = SimLayout::fromText(text);
            CHECK(simStorageOf(layout).text() == text);   // unbound: already canonical
            const SimFrames f = bindSimLayout(layout, dims);
            const std::string bound = f.storage.text();
            CHECK(bound.find("c=c") == std::string::npos);
            CHECK(bound.find("t=t") == std::string::npos);
            const SimFrames again = bindSimLayout(SimLayout::fromText(bound), dims);
            CHECK(again.storage.text() == bound);
            CHECK(again.angles == f.angles);
            CHECK(again.phases == f.phases);
            CHECK(again.nz == f.nz);
            CHECK(again.tileY == f.tileY);
            CHECK(again.frameOf({1, 2, 0, 0, 0}) == f.frameOf({1, 2, 0, 0, 0}));
        }
    }
    SECTION("the parser names what is wrong") {
        CHECK_THROWS_WITH(SimLayout::fromText(""), ContainsSubstring("empty"));
        CHECK_THROWS_WITH(SimLayout::fromText("yes"), ContainsSubstring("'yes' is not a file axis"));
        CHECK_THROWS_WITH(SimLayout::fromText("z[angle 3]"), ContainsSubstring("expected '=', got '['"));
        CHECK_THROWS_WITH(SimLayout::fromText("q=angle 3"), ContainsSubstring("not a file axis"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[angle 3, z, phase 5]; z=phase 3"), ContainsSubstring("given twice"));
        CHECK_THROWS_WITH(SimLayout::fromText("c=angle 3; z=[angle 3, phase 5]"), ContainsSubstring("angle is assigned twice"));
        CHECK_THROWS_WITH(SimLayout::fromText("c=angle; z=phase 3"), ContainsSubstring("angle on c needs its extent"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[c 2, angle 3, phase 5]"), ContainsSubstring("c can only stand on the c axis"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[z 9]"), ContainsSubstring("names angle or phase"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[angle 3, zz, phase 5]"), ContainsSubstring("'zz' is not an axis"));
        CHECK_THROWS_WITH(SimLayout::fromText("yx=3x3[angle 3, phase 5]"), ContainsSubstring("9 tiles, but angle 3 × phase 5 = 15"));
        CHECK_THROWS_WITH(SimLayout::fromText("yx=angle 3"), ContainsSubstring("brackets"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[angle 0, z, phase 5]"), ContainsSubstring("at least 1"));
        CHECK_THROWS_WITH(SimLayout::fromText("z=[angle 3 | phase 5]"), ContainsSubstring("unexpected '|'"));
    }
    SECTION("equality covers the storage text") {
        CHECK(plain == SimLayout::shorthand(3, 5));
        CHECK(plain != general);
        CHECK(general == SimLayout::fromText("z=[angle 3, z, phase 5]"));
    }
}

TEST_CASE("SIM layout: the three real shapes bind, and the arithmetic is loud when they do not", "[app][sim_layout]") {
    SECTION("cudasirecon raw.tif: 135 sections of 3 angles x 5 phases x 9 z on z") {
        const Dims5 raw{1, 1, 135, 64, 64};
        for (const SimLayout& layout : {SimLayout::shorthand(3, 5), SimLayout::fromText("z=[angle 3, z, phase 5]")}) {
            INFO(layout.text());
            CHECK(simLayoutProblem(layout, raw).empty());
            const SimFrames f = bindSimLayout(layout, raw);
            CHECK(f.angles == 3);
            CHECK(f.phases == 5);
            CHECK(f.nz == 9);
            CHECK(f.channels == 1);
            CHECK(f.times == 1);
            CHECK(f.tileY == 64);
            CHECK(f.tileX == 64);
            CHECK(f.sections() == 135);
            CHECK(f.storage.text() == "z=[angle 3, z 9, phase 5]");   // the remainder filled in
            REQUIRE(f.libraryOrder().has_value());
            CHECK_FALSE(*f.libraryOrder());
            CHECK(f.frameOf({2, 4, 8, 0, 0}) == SimFrames::Frame{0, 0, 134, 0, 0});
            CHECK(f.logicalOf({0, 0, 134, 0, 0}) == SimFrames::Logical{2, 4, 8, 0, 0});
        }
        // the message quotes the numbers: the sections and the product that has to divide them
        Dims5 bad = raw;
        bad.z = 134;
        CHECK(simLayoutProblem(SimLayout::shorthand(3, 5), bad) == "z holds 134 sections, not a multiple of angle 3 × phase 5 = 15.");
        CHECK_THROWS_WITH(bindSimLayout(SimLayout::fromText("z=[angle 3, z, phase 5]"), bad), ContainsSubstring("134 sections, not a multiple of angle 3 × phase 5 = 15"));
        // every extent given: the product has to be the length
        CHECK(simLayoutProblem(SimLayout::fromText("z=[angle 3, z 8, phase 5]"), raw) == "z holds 135 sections, but the layout needs angle 3 × z 8 × phase 5 = 120.");
    }
    // A 3-channel, 3-section stack with the angle on c and the phase on z. WHICH axis mcSIM's
    // synthetic_microtubules.tiff actually writes the angle on is NOT settled: both orientations
    // bind and reconstruct, and the measured modulation depths (0.05 one way, 0.13-0.17 the other)
    // favour the phase on c, which is the opposite of what this file was first named for. The file
    // does not record its writer's axis order. The arithmetic below is what is being tested, and it
    // holds either way; do not read the section name as a fact about that dataset.
    SECTION("c3 z3: the angle on the channel axis, the phase on z") {
        const Dims5 mc{3, 1, 3, 2048, 2048};
        const SimLayout layout = SimLayout::fromText("c=angle 3; z=phase 3");
        CHECK(layout.ndirs == 3);
        CHECK(layout.nphases == 3);
        CHECK(simLayoutProblem(layout, mc).empty());
        const SimFrames f = bindSimLayout(layout, mc);
        CHECK(f.angles == 3);
        CHECK(f.phases == 3);
        CHECK(f.nz == 1);
        CHECK(f.channels == 1);   // the channels were the angles
        CHECK(f.times == 1);
        CHECK(f.tileY == 2048);
        CHECK(f.tileX == 2048);
        CHECK_FALSE(f.libraryOrder().has_value());   // has to be gathered
        CHECK(f.frameOf({2, 1, 0, 0, 0}) == SimFrames::Frame{2, 0, 1, 0, 0});
        CHECK(f.logicalOf({1, 0, 2, 0, 0}) == SimFrames::Logical{1, 2, 0, 0, 0});
        // c3 cannot be 4 angles, and the identity says so with both numbers
        CHECK(simLayoutProblem(SimLayout::fromText("c=angle 4; z=phase 3"), mc) == "c holds 3 channels, but the layout needs angle 4 = 4.");
        // the real channels left over: 6 channels of 3 angles each
        const SimFrames two = bindSimLayout(SimLayout::fromText("c=[c, angle 3]; z=phase 3"), Dims5{6, 1, 3, 64, 64});
        CHECK(two.channels == 2);
        CHECK(two.frameOf({1, 2, 0, 1, 0}) == SimFrames::Frame{4, 0, 2, 0, 0});   // channel 1, angle 1 -> c index 1 * 3 + 1
        CHECK(two.logicalOf({5, 0, 0, 0, 0}) == SimFrames::Logical{2, 0, 0, 1, 0});
        CHECK(simLayoutProblem(SimLayout::fromText("c=[c, angle 4]; z=phase 3"), Dims5{6, 1, 3, 64, 64}) == "c holds 6 channels, not a multiple of angle 4 = 4.");
    }
    SECTION("OpenSIM sim01z4.tif: a 3 x 3 montage of 512 x 512 frames in a 1536 x 1536 plane") {
        const Dims5 open{1, 1, 1, 1536, 1536};
        const SimLayout layout = SimLayout::fromText("yx=3x3[angle 3, phase 3]");
        CHECK(simLayoutProblem(layout, open).empty());
        const SimFrames f = bindSimLayout(layout, open);
        CHECK(f.angles == 3);
        CHECK(f.phases == 3);
        CHECK(f.nz == 1);
        CHECK(f.rows == 3);
        CHECK(f.cols == 3);
        CHECK(f.tileY == 512);
        CHECK(f.tileX == 512);
        CHECK_FALSE(f.libraryOrder().has_value());
        // angle down the rows, phase across the columns: tile 7 is (row 2, col 1)
        CHECK(f.frameOf({2, 1, 0, 0, 0}) == SimFrames::Frame{0, 0, 0, 2, 1});
        CHECK(f.logicalOf({0, 0, 0, 1, 2}) == SimFrames::Logical{1, 2, 0, 0, 0});
        // the grid has to divide the plane
        CHECK(simLayoutProblem(layout, Dims5{1, 1, 1, 1535, 1536}) == "y holds 1535 rows, not a multiple of 3 tile rows.");
        CHECK(simLayoutProblem(layout, Dims5{1, 1, 1, 1536, 1000}) == "x holds 1000 columns, not a multiple of 3 tile columns.");
        // a 5 x 3 grid read row by row with the phase outermost
        const SimFrames wide = bindSimLayout(SimLayout::fromText("yx=5x3[phase 5, angle 3]"), Dims5{1, 1, 4, 500, 300});
        CHECK(wide.tileY == 100);
        CHECK(wide.tileX == 100);
        CHECK(wide.nz == 4);   // the z axis is itself
        CHECK(wide.frameOf({1, 4, 3, 0, 0}) == SimFrames::Frame{0, 0, 3, 4, 1});   // tile 4 * 3 + 1 = 13 -> row 4, col 1
        CHECK(wide.logicalOf({0, 0, 3, 4, 1}) == SimFrames::Logical{1, 4, 3, 0, 0});
    }
    SECTION("a frame outside the layout is refused, not wrapped") {
        const SimFrames f = bindSimLayout(SimLayout::shorthand(3, 5), Dims5{1, 1, 135, 64, 64});
        CHECK_THROWS_AS(f.frameOf({3, 0, 0, 0, 0}), std::out_of_range);
        CHECK_THROWS_AS(f.frameOf({0, 5, 0, 0, 0}), std::out_of_range);
        CHECK_THROWS_AS(f.frameOf({0, 0, 9, 0, 0}), std::out_of_range);
        CHECK_THROWS_AS(f.logicalOf({0, 0, 135, 0, 0}), std::out_of_range);
    }
}

TEST_CASE("SIM layout: the z-packed section index is the formula the reconstructor has always used", "[app][sim_layout]") {
    // every (ndirs, nphases, nz, fastSi) a stack has been read with, both
    // through the shorthand and through its text form
    struct Case {
        int ndirs, nphases;
        Index nz;
        bool fastSi;
    };
    for (const Case& c : {Case{3, 5, 9, false}, Case{3, 5, 9, true}, Case{3, 3, 1, false}, Case{3, 3, 1, true}, Case{5, 3, 4, true},
                          Case{1, 3, 7, false}, Case{1, 3, 7, true}, Case{2, 7, 3, false}, Case{4, 2, 2, true}}) {
        const Dims5 dims{1, 1, static_cast<Index>(c.ndirs) * c.nphases * c.nz, 8, 8};
        const SimLayout shorthand = SimLayout::shorthand(c.ndirs, c.nphases, c.fastSi);
        const SimLayout text = SimLayout::fromText(shorthand.text());
        INFO(shorthand.text() << " on " << dims.toString());
        CHECK(text.fastSi == c.fastSi);
        for (const SimLayout& layout : {shorthand, text}) {
            const SimFrames f = bindSimLayout(layout, dims);
            REQUIRE(f.nz == c.nz);
            std::set<Index> seen;
            for (Index d = 0; d < c.ndirs; ++d)
                for (Index z = 0; z < c.nz; ++z)
                    for (Index ph = 0; ph < c.nphases; ++ph) {
                        const Index s = f.sectionIndex(d, ph, z);
                        REQUIRE(s == oldFormula(c.ndirs, c.nphases, c.nz, c.fastSi, d, z, ph));
                        REQUIRE(f.logicalOf({0, 0, s, 0, 0}) == SimFrames::Logical{d, ph, z, 0, 0});
                        seen.insert(s);
                    }
            CHECK(static_cast<Index>(seen.size()) == dims.z);   // a bijection onto the sections
            // the stack is read as stored, in the order the parameters say
            // (both orders coincide when there is one plane or one angle)
            const std::optional<bool> order = f.libraryOrder();
            REQUIRE(order.has_value());
            if (c.nz > 1 && c.ndirs > 1) CHECK(*order == c.fastSi);
        }
    }
    SECTION("an order the reconstructor does not read is gathered") {
        const SimFrames f = bindSimLayout(SimLayout::fromText("z=[phase 5, angle 3, z]"), Dims5{1, 1, 135, 8, 8});
        CHECK_FALSE(f.libraryOrder().has_value());
        CHECK(f.sectionIndex(1, 2, 3) == (2 * 3 + 1) * 9 + 3);
    }
}
