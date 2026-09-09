#include "sirius/real_fft.hpp"
#include "fftw_internal.hpp"
#include "real_fft_backend.hpp"

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Core>
#include <fftw3.h>

namespace sirius {
    namespace {
        struct RealFftwTraits {
            using Plan = fftw_plan;

            static int alignmentOf(double* p) {
                return fftw_alignment_of(p);
            }
            static fftw_plan planR2C(int rank, const int* dims, int howmany,
                                     double* in, int real_dist,
                                     std::complex<double>* out, int complex_dist,
                                     unsigned flags) {
                return fftw_plan_many_dft_r2c(
                    rank, dims, howmany,
                    in, nullptr, 1, real_dist,
                    reinterpret_cast<fftw_complex*>(out), nullptr, 1, complex_dist,
                    flags);
            }
            static fftw_plan planC2R(int rank, const int* dims, int howmany,
                                     std::complex<double>* in, int complex_dist,
                                     double* out, int real_dist,
                                     unsigned flags) {
                return fftw_plan_many_dft_c2r(
                    rank, dims, howmany,
                    reinterpret_cast<fftw_complex*>(in), nullptr, 1, complex_dist,
                    out, nullptr, 1, real_dist,
                    flags);
            }
            static void executeR2C(fftw_plan plan, double* in, std::complex<double>* out) {
                fftw_execute_dft_r2c(plan, in, reinterpret_cast<fftw_complex*>(out));
            }
            static void executeC2R(fftw_plan plan, std::complex<double>* in, double* out) {
                fftw_execute_dft_c2r(plan, reinterpret_cast<fftw_complex*>(in), out);
            }
            static void destroyPlan(fftw_plan plan) {
                fftw_destroy_plan(plan);
            }
        };

        // Destruction is planner state as much as creation (see the note in
        // fft.cpp): serialized on the planner mutex.
        struct RealPlanDeleter {
            void operator()(RealFftwTraits::Plan plan) const {
                if (!plan) return;
                std::lock_guard<std::mutex> lock(detail::fftwPlannerMutex());
                RealFftwTraits::destroyPlan(plan);
            }
        };

        using RealPlanPtr = std::unique_ptr<
            std::remove_pointer_t<RealFftwTraits::Plan>,
            RealPlanDeleter>;

        bool isAlignedForPlan(const void* ptr, int plan_alignment) {
            return RealFftwTraits::alignmentOf(
                       reinterpret_cast<double*>(const_cast<void*>(ptr))) == plan_alignment;
        }

        class FftwRealBackend final : public detail::RealFftBackend {
        public:
            FftwRealBackend(const std::vector<int>& dims, int howmany, PlanRigor rigor) {
                const int real_size = detail::checkedProduct(dims, "RealFFT");
                std::vector<int> complex_dims = dims;
                complex_dims.back() = complex_dims.back() / 2 + 1;
                const int complex_size = detail::checkedProduct(complex_dims, "RealFFT half-complex");
                full_real_size_ = detail::checkedMultiply(real_size, howmany, "RealFFT real");
                full_complex_size_ = detail::checkedMultiply(complex_size, howmany, "RealFFT complex");

                std::unique_ptr<double, detail::FftwTypedFree<double>> buf_in(
                    static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * full_real_size_)));
                std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> buf_out(
                    static_cast<std::complex<double>*>(
                        detail::checkedFftwMalloc(sizeof(std::complex<double>) * full_complex_size_)));
                alignment_ = RealFftwTraits::alignmentOf(buf_in.get());

                const unsigned flags = detail::toFFTWFlag(rigor);
                std::lock_guard<std::mutex> lock(detail::fftwPlannerMutex());
                detail::ensureDoubleThreadsInitializedLocked();

                forward_plan_ = RealPlanPtr(
                    RealFftwTraits::planR2C(
                        static_cast<int>(dims.size()), dims.data(), howmany,
                        buf_in.get(), real_size, buf_out.get(), complex_size, flags));
                inverse_plan_ = RealPlanPtr(
                    RealFftwTraits::planC2R(
                        static_cast<int>(dims.size()), dims.data(), howmany,
                        buf_out.get(), complex_size, buf_in.get(), real_size, flags));

                if (!forward_plan_ || !inverse_plan_)
                    throw std::runtime_error("FFTW failed to create real FFT plan.");
            }

            void rfft(const double* in, std::complex<double>* out, const Stream&) const override {
                double* in_ptr = const_cast<double*>(in);
                const bool aligned =
                    isAlignedForPlan(in_ptr, alignment_) &&
                    isAlignedForPlan(out, alignment_);
                if (aligned) {
                    RealFftwTraits::executeR2C(forward_plan_.get(), in_ptr, out);
                    return;
                }
                std::unique_ptr<double, detail::FftwTypedFree<double>> tmp_in(
                    static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * full_real_size_)));
                std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> tmp_out(
                    static_cast<std::complex<double>*>(
                        detail::checkedFftwMalloc(sizeof(std::complex<double>) * full_complex_size_)));
                std::memcpy(tmp_in.get(), in, sizeof(double) * full_real_size_);
                RealFftwTraits::executeR2C(forward_plan_.get(), tmp_in.get(), tmp_out.get());
                std::memcpy(out, tmp_out.get(), sizeof(std::complex<double>) * full_complex_size_);
            }

            void irfft(const std::complex<double>* in, double* out, const Stream&) const override {
                // FFTW's c2r execution may overwrite its complex input, so copy
                // it even when alignment is otherwise suitable.
                std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> tmp_in(
                    static_cast<std::complex<double>*>(
                        detail::checkedFftwMalloc(sizeof(std::complex<double>) * full_complex_size_)));
                std::memcpy(tmp_in.get(), in, sizeof(std::complex<double>) * full_complex_size_);

                if (isAlignedForPlan(out, alignment_)) {
                    RealFftwTraits::executeC2R(inverse_plan_.get(), tmp_in.get(), out);
                    return;
                }
                std::unique_ptr<double, detail::FftwTypedFree<double>> tmp_out(
                    static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * full_real_size_)));
                RealFftwTraits::executeC2R(inverse_plan_.get(), tmp_in.get(), tmp_out.get());
                std::memcpy(out, tmp_out.get(), sizeof(double) * full_real_size_);
            }

            void scaleReal(double* out, std::size_t n, double scale, const Stream&) const override {
                Eigen::Map<Eigen::VectorXd>(out, static_cast<Eigen::Index>(n)) *= scale;
            }

        private:
            RealPlanPtr forward_plan_;
            RealPlanPtr inverse_plan_;
            int full_real_size_ = 0;
            int full_complex_size_ = 0;
            int alignment_ = 0;
        };
    } // namespace

    namespace detail {
        std::unique_ptr<RealFftBackend> makeFftwRealBackend(const std::vector<int>& dims, int howmany,
                                                            PlanRigor rigor) {
            return std::make_unique<FftwRealBackend>(dims, howmany, rigor);
        }
    } // namespace detail

    struct RealFFT::Impl {
        std::unique_ptr<detail::RealFftBackend> backend;
        std::vector<int> dims;
        Device device;
        int howmany = 1;
        int real_size = 0;
        int complex_size = 0;
        int full_real_size = 0;
        int full_complex_size = 0;
    };

    RealFFT::RealFFT(std::vector<int> dims, int howmany, PlanRigor rigor, Device device)
        : impl_(std::make_unique<Impl>()) {
        if (dims.empty() || dims.size() > 3)
            throw std::invalid_argument("Only ranks 1, 2 and 3 are supported.");
        if (howmany < 1)
            throw std::invalid_argument("howmany must be >= 1");

        const int real_size = detail::checkedProduct(dims, "RealFFT");
        std::vector<int> complex_dims = dims;
        complex_dims.back() = complex_dims.back() / 2 + 1;
        const int complex_size = detail::checkedProduct(complex_dims, "RealFFT half-complex");

        impl_->dims = std::move(dims);
        impl_->device = device;
        impl_->howmany = howmany;
        impl_->real_size = real_size;
        impl_->complex_size = complex_size;
        impl_->full_real_size = detail::checkedMultiply(real_size, howmany, "RealFFT real");
        impl_->full_complex_size = detail::checkedMultiply(complex_size, howmany, "RealFFT complex");

        requireDevice(device);
        if (device.isCuda())
            impl_->backend = detail::makeCufftRealBackend(impl_->dims, howmany, device);
        else
            impl_->backend = detail::makeFftwRealBackend(impl_->dims, howmany, rigor);
    }

    RealFFT::~RealFFT() = default;
    RealFFT::RealFFT(RealFFT&&) noexcept = default;
    RealFFT& RealFFT::operator=(RealFFT&&) noexcept = default;

    Device RealFFT::device() const noexcept { return impl_->device; }
    int RealFFT::rank() const { return static_cast<int>(impl_->dims.size()); }
    int RealFFT::howmany() const { return impl_->howmany; }
    int RealFFT::realSize() const { return impl_->real_size; }
    int RealFFT::complexSize() const { return impl_->complex_size; }
    int RealFFT::fullRealSize() const { return impl_->full_real_size; }
    int RealFFT::fullComplexSize() const { return impl_->full_complex_size; }
    const std::vector<int>& RealFFT::dims() const { return impl_->dims; }

    void RealFFT::rfft(const Real* in, Complex* out, const Stream& stream) const {
        impl_->backend->rfft(in, out, stream);
    }

    void RealFFT::irfft(const Complex* in, Real* out, bool normalize, const Stream& stream) const {
        impl_->backend->irfft(in, out, stream);
        if (normalize)
            impl_->backend->scaleReal(out, static_cast<std::size_t>(impl_->full_real_size),
                                      1.0 / static_cast<double>(impl_->real_size), stream);
    }

    namespace {
        void checkTensorSize(const char* what, Eigen::Index have, int want) {
            if (have != static_cast<Eigen::Index>(want))
                throw std::invalid_argument(std::string("RealFFT: ") + what + " tensor holds " + std::to_string(have) +
                                            " elements, the plan needs " + std::to_string(want));
        }
    } // namespace

    template <int Rank>
    void RealFFT::rfft(const TensorXr<double, Rank>& in,
                       TensorXc<double, Rank>& out) const {
        // The pointer API trusts its caller; the tensor API knows the sizes.
        checkTensorSize("input", in.size(), impl_->full_real_size);
        checkTensorSize("output", out.size(), impl_->full_complex_size);
        rfft(in.data(), out.data());
    }

    template <int Rank>
    void RealFFT::irfft(const TensorXc<double, Rank>& in,
                        TensorXr<double, Rank>& out,
                        bool normalize) const {
        checkTensorSize("input", in.size(), impl_->full_complex_size);
        checkTensorSize("output", out.size(), impl_->full_real_size);
        irfft(in.data(), out.data(), normalize);
    }

    template void RealFFT::rfft(const TensorXr<double, 1>&, TensorXc<double, 1>&) const;
    template void RealFFT::rfft(const TensorXr<double, 2>&, TensorXc<double, 2>&) const;
    template void RealFFT::rfft(const TensorXr<double, 3>&, TensorXc<double, 3>&) const;
    template void RealFFT::irfft(const TensorXc<double, 1>&, TensorXr<double, 1>&, bool) const;
    template void RealFFT::irfft(const TensorXc<double, 2>&, TensorXr<double, 2>&, bool) const;
    template void RealFFT::irfft(const TensorXc<double, 3>&, TensorXr<double, 3>&, bool) const;

#ifndef SIRIUS_HAS_CUDA
    namespace detail {
        std::unique_ptr<RealFftBackend> makeCufftRealBackend(const std::vector<int>&, int, Device device) {
            throw std::runtime_error("SIRIUS was built without CUDA support; cannot plan a real FFT on " +
                                     toString(device));
        }
    } // namespace detail
#endif

} // namespace sirius
