// Buffer / BufferView / Shape / Stream tests. The CUDA sections run only when
// a GPU is usable at run time (they SKIP otherwise), so the same binary is
// meaningful on CPU-only CI and on a GPU workstation.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <complex>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/errors.hpp"
#include "sirius/tiff_io.hpp"

using namespace sirius;

namespace {
    // Returns the CUDA device to test on, or skips the enclosing test.
    Device gpuOrSkip() {
        if (!cudaAvailable()) SKIP("no CUDA device available");
        return Device::cuda(0);
    }

    template <typename T>
    std::vector<T> iota(Index n, T start = T{}) {
        std::vector<T> v(static_cast<std::size_t>(n));
        std::iota(v.begin(), v.end(), start);
        return v;
    }

    template <typename T>
    Buffer<T> hostIota(Shape shape) {
        Buffer<T> b(shape);
        for (Index i = 0; i < b.size(); ++i) b.data()[i] = static_cast<T>(i);
        return b;
    }
} // namespace

// -----------------------------------------------------------------------
// Shape
// -----------------------------------------------------------------------

TEST_CASE("Shape basics", "[buffer][shape]") {
    Shape s{3, 4, 5};
    REQUIRE(s.rank() == 3);
    REQUIRE(s[0] == 3);
    REQUIRE(s[2] == 5);
    REQUIRE(s.numel() == 60);
    REQUIRE(s.toString() == "(3, 4, 5)");
    REQUIRE(s == Shape{3, 4, 5});
    REQUIRE(s != Shape{3, 4});

    SECTION("empty shape has no elements") {
        Shape e;
        REQUIRE(e.rank() == 0);
        REQUIRE(e.numel() == 0);
        REQUIRE(e.empty());
    }
    SECTION("zero-length dimension") {
        REQUIRE(Shape{0, 5}.numel() == 0);
    }
    SECTION("a shape mismatch is a ShapeError without ceasing to be an invalid_argument") {
        Buffer<float> a(Shape{2, 3}), b(Shape{3, 2});
        REQUIRE_THROWS_AS(copy(a, b), ShapeError);
        REQUIRE_THROWS_AS(copy(a, b), std::invalid_argument);
        REQUIRE_THROWS_AS(copy(a, b), std::exception);
        REQUIRE_THROWS_WITH(copy(a, b), "copy: shape mismatch (2, 3) vs (3, 2)");
    }
    SECTION("rank limit and negative dims are rejected") {
        REQUIRE_THROWS_AS(Shape({1, 1, 1, 1, 1}), std::invalid_argument);
        REQUIRE_THROWS_AS(Shape({1, -1}), std::invalid_argument);
    }
    SECTION("asStack collapses leading dimensions") {
        REQUIRE(Shape{7}.asStack() == Shape{1, 1, 7});
        REQUIRE(Shape{4, 7}.asStack() == Shape{1, 4, 7});
        REQUIRE(Shape{2, 4, 7}.asStack() == Shape{2, 4, 7});
        REQUIRE(Shape{3, 2, 4, 7}.asStack() == Shape{6, 4, 7});
    }
    SECTION("from an Eigen tensor") {
        ImageStack<float> t(2, 3, 4);
        REQUIRE(Shape(t) == Shape{2, 3, 4});
    }
}

// -----------------------------------------------------------------------
// Host buffers
// -----------------------------------------------------------------------

TEST_CASE("Host Buffer allocation and accessors", "[buffer]") {
    Buffer<float> b(Shape{2, 3});
    REQUIRE(b.device() == Device::cpu());
    REQUIRE(b.size() == 6);
    REQUIRE(b.bytes() == 6 * sizeof(float));
    REQUIRE(b.rank() == 2);
    REQUIRE(b.dim(1) == 3);
    REQUIRE_FALSE(b.empty());
    REQUIRE_FALSE(b.pinned());
    // 64-byte alignment for SIMD
    REQUIRE(reinterpret_cast<std::uintptr_t>(b.data()) % 64 == 0);

    SECTION("default constructed buffer is empty") {
        Buffer<int> e;
        REQUIRE(e.empty());
        REQUIRE(e.size() == 0);
        REQUIRE(e.data() == nullptr);
    }
    SECTION("zero-sized shape allocates nothing") {
        Buffer<int> z(Shape{0, 4});
        REQUIRE(z.empty());
        REQUIRE(z.size() == 0);
    }
    SECTION("move semantics transfer ownership") {
        float* p = b.data();
        Buffer<float> m = std::move(b);
        REQUIRE(m.data() == p);
        REQUIRE(m.shape() == Shape{2, 3});
        REQUIRE(b.empty());

        Buffer<float> n;
        n = std::move(m);
        REQUIRE(n.data() == p);
        REQUIRE(m.empty());
    }
    SECTION("reset frees") {
        b.reset();
        REQUIRE(b.empty());
        REQUIRE(b.shape().rank() == 0);
    }
    SECTION("reshape keeps the memory") {
        float* p = b.data();
        b.reshape(Shape{6});
        REQUIRE(b.data() == p);
        REQUIRE(b.rank() == 1);
        REQUIRE_THROWS_AS(b.reshape(Shape{7}), std::invalid_argument);
    }
}

TEST_CASE("Buffer::adopt wraps external memory and runs the deleter", "[buffer]") {
    std::vector<int> storage(10, 7);
    bool deleted = false;
    {
        auto b = Buffer<int>::adopt(storage.data(), Shape{10}, Device::cpu(),
                                    [&](int*) { deleted = true; });
        REQUIRE(b.data() == storage.data());
        REQUIRE(b.data()[3] == 7);
        REQUIRE_FALSE(deleted);
    }
    REQUIRE(deleted);
}

TEST_CASE("fill, copy and clone on the host", "[buffer]") {
    Buffer<std::uint16_t> a(Shape{4, 5});
    fill(a, std::uint16_t{42});
    for (Index i = 0; i < a.size(); ++i) REQUIRE(a.data()[i] == 42);

    Buffer<std::uint16_t> c = a.clone();
    REQUIRE(c.data() != a.data());
    REQUIRE(c.shape() == a.shape());
    REQUIRE(c.data()[19] == 42);

    Buffer<std::uint16_t> d(Shape{4, 5});
    copy(a, d);
    REQUIRE(d.data()[0] == 42);

    SECTION("copy rejects shape mismatches") {
        Buffer<std::uint16_t> wrong(Shape{5, 4});
        REQUIRE_THROWS_AS(copy(a, wrong), std::invalid_argument);
    }
    SECTION("to(cpu) is a clone") {
        auto e = a.to(Device::cpu());
        REQUIRE(e.data() != a.data());
        REQUIRE(e.data()[7] == 42);
    }
}

TEST_CASE("convert casts between pixel types on the host", "[buffer]") {
    auto src = hostIota<std::uint16_t>(Shape{3, 4});
    Buffer<float> dst(Shape{3, 4});
    convert(src, dst);
    for (Index i = 0; i < src.size(); ++i) REQUIRE(dst.data()[i] == static_cast<float>(i));

    SECTION("narrowing conversions truncate in range and saturate out of it") {
        // static_cast is undefined for a float outside the integer's range
        // (and wraps on MSVC) while the GPU's cvt saturates: one rule for both
        Buffer<double> d(Shape{6});
        d.data()[0] = 3.9;
        d.data()[1] = -2.5;
        d.data()[2] = 300.0;
        d.data()[3] = -300.0;
        d.data()[4] = std::numeric_limits<double>::quiet_NaN();
        d.data()[5] = std::numeric_limits<double>::infinity();
        Buffer<std::int8_t> i8(Shape{6});
        convert(d, i8);
        REQUIRE(i8.data()[0] == 3);
        REQUIRE(i8.data()[1] == -2);
        REQUIRE(i8.data()[2] == 127);
        REQUIRE(i8.data()[3] == -128);
        REQUIRE(i8.data()[4] == 0);
        REQUIRE(i8.data()[5] == 127);
        Buffer<float> f(Shape{3});
        f.data()[0] = -1.0f;
        f.data()[1] = 70000.0f;
        f.data()[2] = 65535.0f;
        Buffer<std::uint16_t> u16(Shape{3});
        convert(f, u16);
        REQUIRE(u16.data()[0] == 0);
        REQUIRE(u16.data()[1] == 65535);
        REQUIRE(u16.data()[2] == 65535);
    }
    SECTION("same type is a copy") {
        Buffer<std::uint16_t> same(Shape{3, 4});
        convert(src, same);
        REQUIRE(same.data()[11] == 11);
    }
    SECTION("mismatched shapes are rejected") {
        Buffer<float> bad(Shape{4, 3});
        REQUIRE_THROWS_AS(convert(src, bad), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Views
// -----------------------------------------------------------------------

TEST_CASE("BufferView slicing and reshaping", "[buffer][view]") {
    auto b = hostIota<int>(Shape{3, 2, 2});   // pages of 4 elements
    BufferView<int> v = b;
    REQUIRE(v.size() == 12);

    auto page1 = v.slice(1, 1);
    REQUIRE(page1.shape() == Shape{1, 2, 2});
    REQUIRE(page1.data() == b.data() + 4);
    REQUIRE(page1.data()[0] == 4);

    auto tail = v.slice(1, 2);
    REQUIRE(tail.shape() == Shape{2, 2, 2});
    REQUIRE(tail.data()[7] == 11);

    REQUIRE_THROWS_AS(v.slice(2, 2), std::out_of_range);

    auto flat = v.reshape(Shape{12});
    REQUIRE(flat.rank() == 1);
    REQUIRE_THROWS_AS(v.reshape(Shape{13}), std::invalid_argument);

    SECTION("const conversion") {
        BufferView<const int> cv = v;
        REQUIRE(cv.data() == v.data());
        const Buffer<int>& cb = b;
        BufferView<const int> cv2 = cb;
        REQUIRE(cv2.size() == 12);
    }
}

TEST_CASE("Eigen interop is zero-copy", "[buffer][eigen]") {
    ImageStack<float> stack(2, 3, 4);
    for (Eigen::Index i = 0; i < stack.size(); ++i) stack.data()[i] = static_cast<float>(i);

    auto view = toView(stack);
    REQUIRE(view.data() == stack.data());
    REQUIRE(view.shape() == Shape{2, 3, 4});
    REQUIRE(view.device() == Device::cpu());

    auto cview = toConstView(stack);
    REQUIRE(cview.data() == stack.data());

    // copy from an Eigen tensor into a Buffer and back
    Buffer<float> buf(Shape{2, 3, 4});
    copy(stack, buf);
    REQUIRE(buf.data()[23] == 23.0f);

    auto map = asTensor<float, 3>(buf.view());
    REQUIRE(map.data() == buf.data());
    REQUIRE(map(1, 2, 3) == 23.0f);
    map(0, 0, 0) = -1.0f;
    REQUIRE(buf.data()[0] == -1.0f);

    auto back = toEigen<3>(buf);
    REQUIRE(back.dimension(2) == 4);
    REQUIRE(back(0, 0, 0) == -1.0f);

    REQUIRE_THROWS_AS((asTensor<float, 2>(buf.view())), std::invalid_argument);
}

// -----------------------------------------------------------------------
// Device queries (valid on every build)
// -----------------------------------------------------------------------

TEST_CASE("Device values and queries", "[device]") {
    REQUIRE(Device::cpu().isCpu());
    REQUIRE(Device::cuda(1).isCuda());
    REQUIRE(Device::cuda(1).index == 1);
    REQUIRE(Device::cpu() == Device::cpu());
    REQUIRE(Device::cuda(0) != Device::cuda(1));
    REQUIRE(Device::cpu() != Device::cuda(0));
    REQUIRE(toString(Device::cpu()) == "cpu");
    REQUIRE(toString(Device::cuda(2)) == "cuda:2");

    REQUIRE(cudaDeviceCount() >= 0);
    REQUIRE(cudaAvailable() == (cudaDeviceCount() > 0));
    if (!builtWithCuda()) REQUIRE_FALSE(cudaAvailable());

    REQUIRE_NOTHROW(requireDevice(Device::cpu()));
    REQUIRE_THROWS(requireDevice(Device::cuda(cudaDeviceCount())));   // one past the last
    REQUIRE_THROWS(deviceProperties(Device::cpu()));

    Stream s;   // CPU stream is a no-op
    REQUIRE(s.device().isCpu());
    REQUIRE(s.isNull());
    REQUIRE_NOTHROW(s.synchronize());
    Event e;
    REQUIRE(e.ready());
    REQUIRE_NOTHROW(e.record(s));
    REQUIRE_NOTHROW(e.wait(s));
}

TEST_CASE("CUDA device requested on an unavailable build or machine throws", "[device]") {
    if (cudaAvailable()) SKIP("a CUDA device is available; the failure path does not apply");
    REQUIRE_THROWS_AS(Buffer<float>(Shape{4}, Device::cuda(0)), std::runtime_error);
    REQUIRE_THROWS_AS(Stream(Device::cuda(0)), std::runtime_error);
}

// -----------------------------------------------------------------------
// CUDA
// -----------------------------------------------------------------------

TEST_CASE("Device buffers round-trip through the GPU", "[buffer][cuda]") {
    const Device gpu = gpuOrSkip();
    const auto props = deviceProperties(gpu);
    INFO("GPU: " << props.name << " cc " << props.computeMajor << "." << props.computeMinor);
    REQUIRE(props.totalMemoryBytes > 0);

    auto host = hostIota<std::uint16_t>(Shape{3, 64, 64});

    SECTION("host -> device -> host on the null stream") {
        Buffer<std::uint16_t> dev = toDevice(host, gpu);
        REQUIRE(dev.device() == gpu);
        REQUIRE(dev.shape() == host.shape());
        Buffer<std::uint16_t> back = dev.to(Device::cpu());
        Stream::null().synchronize();
        for (Index i = 0; i < host.size(); ++i) REQUIRE(back.data()[i] == host.data()[i]);
    }

    SECTION("asynchronous path with pinned memory and a stream") {
        Stream stream(gpu);
        REQUIRE(stream.device() == gpu);
        REQUIRE_FALSE(stream.isNull());

        Buffer<std::uint16_t> pinned(host.shape(), Device::cpu(), HostMemory::Pinned);
        REQUIRE(pinned.pinned());
        copy(host, pinned);

        Buffer<std::uint16_t> dev(host.shape(), gpu, HostMemory::Pageable, stream);
        copy(pinned, dev, stream);

        Buffer<std::uint16_t> back(host.shape(), Device::cpu(), HostMemory::Pinned);
        copy(dev, back, stream);
        Event done;
        done.record(stream);
        done.synchronize();
        REQUIRE(done.ready());
        for (Index i = 0; i < host.size(); ++i) REQUIRE(back.data()[i] == host.data()[i]);
    }

    SECTION("device -> device copy and clone") {
        Buffer<std::uint16_t> dev = toDevice(host, gpu);
        Buffer<std::uint16_t> dev2 = dev.clone();
        REQUIRE(dev2.device() == gpu);
        REQUIRE(dev2.data() != dev.data());
        auto back = toEigen<3>(dev2);
        REQUIRE(back(2, 63, 63) == host.data()[host.size() - 1]);
    }

    SECTION("fill on the device") {
        Buffer<float> dev(Shape{1000}, gpu);
        fill(dev, 2.5f);
        auto back = toEigen<1>(dev);
        for (Index i = 0; i < 1000; ++i) REQUIRE(back(i) == 2.5f);

        Buffer<std::complex<double>> cdev(Shape{10}, gpu);
        fill(cdev, std::complex<double>(1.0, -2.0));
        auto cback = toEigen<1>(cdev);
        REQUIRE(cback(9) == std::complex<double>(1.0, -2.0));
    }

    SECTION("convert on the device matches the host") {
        Buffer<std::uint16_t> dev = toDevice(host, gpu);
        Buffer<float> devF(host.shape(), gpu);
        convert(dev, devF);
        Buffer<float> hostF(host.shape());
        convert(host, hostF);
        auto back = toEigen<3>(devF);
        for (Index i = 0; i < host.size(); ++i) REQUIRE(back.data()[i] == hostF.data()[i]);

        Buffer<std::int8_t> devI8(host.shape(), gpu);
        convert(devF, devI8);
        auto backI8 = toEigen<3>(devI8);
        REQUIRE(backI8.data()[300] == static_cast<std::int8_t>(300));
    }

    SECTION("views of device memory cannot be mapped as Eigen tensors") {
        Buffer<std::uint16_t> dev = toDevice(host, gpu);
        REQUIRE_THROWS_AS((asTensor<std::uint16_t, 3>(dev.view())), std::runtime_error);
    }
}

TEST_CASE("A stream on another device is rejected", "[buffer][cuda][multigpu]") {
    gpuOrSkip();
    if (cudaDeviceCount() < 2) SKIP("needs two GPUs");
    Stream other(Device::cuda(1));
    Buffer<std::uint16_t> dev(Shape{16}, Device::cuda(0));
    REQUIRE_THROWS_AS(fill(dev, std::uint16_t{1}, other), std::runtime_error);
}

TEST_CASE("Repeated device allocations are pooled", "[buffer][cuda]") {
    const Device gpu = gpuOrSkip();
    Stream stream(gpu);
    // Allocate/free in a loop: with the memory pool this is cheap and, more
    // importantly, must not leak or error.
    for (int i = 0; i < 200; ++i) {
        Buffer<float> b(Shape{1 << 16}, gpu, HostMemory::Pageable, stream);
        fill(b, static_cast<float>(i), stream);
    }
    stream.synchronize();
    SUCCEED();
}

TEST_CASE("Buffer outlives the stream it was allocated on", "[buffer][cuda]") {
    const Device gpu = gpuOrSkip();
    Buffer<int> dev;
    {
        Stream scratch(gpu);
        dev = Buffer<int>(Shape{256}, gpu, HostMemory::Pageable, scratch);
        fill(dev, 9, scratch);
        scratch.synchronize();
    }   // stream destroyed here; the buffer must still be valid and freeable
    auto back = toEigen<1>(dev);
    REQUIRE(back(255) == 9);
    dev.reset();
    SUCCEED();
}

TEST_CASE("Stream::null().synchronize() waits for legacy-stream work", "[buffer][cuda]") {
    const Device gpu = gpuOrSkip();
    // Work enqueued through the null stream runs on the device's legacy
    // default stream; a pinned destination makes the copy back truly
    // asynchronous, so only a real synchronization guarantees fresh bytes.
    auto host = hostIota<float>(Shape{1 << 20});
    Buffer<float> dev = toDevice(host, gpu);
    Buffer<float> pinned(host.shape(), Device::cpu(), HostMemory::Pinned);
    fill(pinned, -1.0f);
    for (int round = 0; round < 3; ++round) {
        fill(dev, static_cast<float>(round));   // kernel on the legacy stream
        copy(dev, pinned);                       // D2H into pinned memory: asynchronous
        Stream::null().synchronize();
        for (Index i = 0; i < pinned.size(); i += 4097)
            REQUIRE(pinned.data()[i] == static_cast<float>(round));
    }
}

TEST_CASE("Shape and Buffer refuse element counts that overflow", "[buffer][shape][overflow]") {
    // 2^80 elements: the product wraps a 64-bit Index, so it must throw
    // rather than allocate a tiny buffer for a huge shape
    const Shape huge{1 << 20, 1 << 20, 1 << 20, 1 << 20};
    CHECK_THROWS_AS(huge.numel(), std::overflow_error);
    CHECK_THROWS_AS(Buffer<float>(huge), std::overflow_error);
    CHECK_FALSE(huge.empty());   // decided without the product
    try {
        (void)huge.numel();
    } catch (const std::overflow_error& e) {
        CHECK(std::string(e.what()).find("1048576 x 1048576 x 1048576 x 1048576") != std::string::npos);
    }
    // 2^61 elements fit an Index but 2^64 bytes do not
    const Shape wide{1 << 20, 1 << 20, 1 << 21};
    CHECK(wide.numel() == (Index{1} << 61));
    CHECK_THROWS_AS(Buffer<double>(wide), std::overflow_error);
    CHECK_THROWS_AS(BufferView<double>(nullptr, wide, Device::cpu()).bytes(), std::overflow_error);
    // ordinary shapes are unaffected
    CHECK(Shape{3, 4, 5}.numel() == 60);
    CHECK(Shape{3, 0, 5}.empty());
    CHECK(Buffer<std::uint16_t>(Shape{3, 4, 5}).bytes() == 120);
}

TEST_CASE("BufferView::slice refuses a count that would wrap past the end", "[buffer][view]") {
    Buffer<float> b(Shape{4, 2});
    // first + count overflowed to a negative number and passed the old check
    CHECK_THROWS_AS(b.view().slice(2, std::numeric_limits<Index>::max()), std::out_of_range);
    CHECK_THROWS_AS(b.view().slice(4, 1), std::out_of_range);
    CHECK(b.view().slice(4, 0).dim(0) == 0);
    CHECK(b.view().slice(1, 3).dim(0) == 3);
}
