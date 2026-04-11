#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <fftw3.h>

#include "sirius/fft_buffers.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// FFTWBuffer1D — construction
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D construction", "[fft_buffers][1d]") {
    SECTION("Valid sizes construct without throwing") {
        auto n = GENERATE(1, 2, 64, 1024);
        REQUIRE_NOTHROW(FFTWBuffer1D(n));
    }

    SECTION("Zero size throws") {
        REQUIRE_THROWS_AS(FFTWBuffer1D(0), std::invalid_argument);
    }

    SECTION("Negative size throws") {
        REQUIRE_THROWS_AS(FFTWBuffer1D(-1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer1D — accessors
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D accessors", "[fft_buffers][1d]") {
    FFTWBuffer1D buf(64);

    SECTION("size() returns constructed size") {
        REQUIRE(buf.size() == 64);
    }

    SECTION("data() is non-null") {
        REQUIRE(buf.data() != nullptr);
    }

    SECTION("const data() is non-null") {
        const FFTWBuffer1D& cbuf = buf;
        REQUIRE(cbuf.data() != nullptr);
    }

    SECTION("as_eigen() maps correct size") {
        REQUIRE(buf.as_eigen().size() == 64);
    }

    SECTION("as_eigen() aliases the same memory as data()") {
        auto m = buf.as_eigen();
        REQUIRE(reinterpret_cast<fftw_complex*>(m.data()) == buf.data());
    }

    SECTION("writes via as_eigen() are visible through data()") {
        buf.as_eigen()[3] = std::complex<double>(1.5, -2.5);
        auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
        REQUIRE(raw[3] == std::complex<double>(1.5, -2.5));
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer1D — move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D move semantics", "[fft_buffers][1d]") {
    SECTION("Move constructor transfers ownership and zeros source") {
        FFTWBuffer1D a(64);
        fftw_complex* ptr = a.data();

        FFTWBuffer1D b(std::move(a));

        REQUIRE(b.data() == ptr);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.size() == 0);
    }

    SECTION("Move assignment transfers ownership and zeros source") {
        FFTWBuffer1D a(64);
        FFTWBuffer1D b(32);
        fftw_complex* ptr = a.data();

        b = std::move(a);

        REQUIRE(b.data() == ptr);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.size() == 0);
    }

    SECTION("Self-assignment is a no-op") {
        FFTWBuffer1D a(64);
        fftw_complex* ptr = a.data();

        a = std::move(a);

        REQUIRE(a.data() == ptr);
        REQUIRE(a.size() == 64);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — construction
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D construction", "[fft_buffers][2d]") {
    SECTION("Valid dimensions construct without throwing") {
        REQUIRE_NOTHROW(FFTWBuffer2D(1, 1));
        REQUIRE_NOTHROW(FFTWBuffer2D(4, 8));
        REQUIRE_NOTHROW(FFTWBuffer2D(128, 256));
    }

    SECTION("Zero rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(0, 8), std::invalid_argument);
    }

    SECTION("Zero cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(4, 0), std::invalid_argument);
    }

    SECTION("Negative rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(-1, 8), std::invalid_argument);
    }

    SECTION("Negative cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(4, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — accessors
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D accessors", "[fft_buffers][2d]") {
    FFTWBuffer2D buf(4, 8);

    SECTION("rows() and cols() return constructed values") {
        REQUIRE(buf.rows() == 4);
        REQUIRE(buf.cols() == 8);
    }

    SECTION("size() equals rows * cols") {
        REQUIRE(buf.size() == 32);
    }

    SECTION("data() is non-null") {
        REQUIRE(buf.data() != nullptr);
    }

    SECTION("const data() is non-null") {
        const FFTWBuffer2D& cbuf = buf;
        REQUIRE(cbuf.data() != nullptr);
    }

    SECTION("as_eigen() maps correct dimensions") {
        auto m = buf.as_eigen();
        REQUIRE(m.rows() == 4);
        REQUIRE(m.cols() == 8);
    }

    SECTION("as_eigen() aliases the same memory as data()") {
        auto m = buf.as_eigen();
        REQUIRE(reinterpret_cast<fftw_complex*>(m.data()) == buf.data());
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — row-major memory layout
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D as_eigen() uses row-major layout matching FFTW C-order", "[fft_buffers][2d]") {
    // FFTW 2D plans assume C-order (row-major): element (r, c) is at offset r*cols + c.
    // as_eigen() must map with RowMajor so Eigen and FFTW agree on element positions.
    FFTWBuffer2D buf(3, 4);
    auto m = buf.as_eigen();

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            m(r, c) = std::complex<double>(r * 10.0 + c, 0.0);

    auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            REQUIRE(raw[r * 4 + c] == std::complex<double>(r * 10.0 + c, 0.0));
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D move semantics", "[fft_buffers][2d]") {
    SECTION("Move constructor transfers ownership, dimensions, and size") {
        FFTWBuffer2D a(4, 8);
        fftw_complex* ptr = a.data();

        FFTWBuffer2D b(std::move(a));

        REQUIRE(b.data() == ptr);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 32);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Move assignment transfers ownership, dimensions, and size") {
        FFTWBuffer2D a(4, 8);
        FFTWBuffer2D b(2, 2);
        fftw_complex* ptr = a.data();

        b = std::move(a);

        REQUIRE(b.data() == ptr);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 32);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Self-assignment is a no-op") {
        FFTWBuffer2D a(4, 8);
        fftw_complex* ptr = a.data();

        a = std::move(a);

        REQUIRE(a.data() == ptr);
        REQUIRE(a.rows() == 4);
        REQUIRE(a.cols() == 8);
        REQUIRE(a.size() == 32);
    }
}
