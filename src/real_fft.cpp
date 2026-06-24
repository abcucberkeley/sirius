#include "sirius/real_fft.hpp"
#include "fftw_internal.hpp"

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

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

        struct RealPlanDeleter {
            void operator()(RealFftwTraits::Plan plan) const {
                if (plan) RealFftwTraits::destroyPlan(plan);
            }
        };

        using RealPlanPtr = std::unique_ptr<
            std::remove_pointer_t<RealFftwTraits::Plan>,
            RealPlanDeleter>;

        bool isAlignedForPlan(const void* ptr, int plan_alignment) {
            return RealFftwTraits::alignmentOf(
                reinterpret_cast<double*>(const_cast<void*>(ptr))) == plan_alignment;
        }

        void execute_rfft_safe(RealFftwTraits::Plan plan,
                               int plan_alignment,
                               int full_real_size,
                               int full_complex_size,
                               const double* in,
                               std::complex<double>* out) {
            double* in_ptr = const_cast<double*>(in);
            const bool aligned =
                isAlignedForPlan(in_ptr, plan_alignment) &&
                isAlignedForPlan(out, plan_alignment);

            if (aligned) {
                RealFftwTraits::executeR2C(plan, in_ptr, out);
                return;
            }

            std::unique_ptr<double, detail::FftwTypedFree<double>> tmp_in(
                static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * full_real_size)));
            std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> tmp_out(
                static_cast<std::complex<double>*>(
                    detail::checkedFftwMalloc(sizeof(std::complex<double>) * full_complex_size)));
            std::memcpy(tmp_in.get(), in, sizeof(double) * full_real_size);
            RealFftwTraits::executeR2C(plan, tmp_in.get(), tmp_out.get());
            std::memcpy(out, tmp_out.get(), sizeof(std::complex<double>) * full_complex_size);
        }

        void execute_irfft_safe(RealFftwTraits::Plan plan,
                                int plan_alignment,
                                int full_real_size,
                                int full_complex_size,
                                const std::complex<double>* in,
                                double* out) {
            // FFTW's c2r execution may overwrite its complex input, so copy it
            // even when alignment is otherwise suitable.
            std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> tmp_in(
                static_cast<std::complex<double>*>(
                    detail::checkedFftwMalloc(sizeof(std::complex<double>) * full_complex_size)));
            std::memcpy(tmp_in.get(), in, sizeof(std::complex<double>) * full_complex_size);

            if (isAlignedForPlan(out, plan_alignment)) {
                RealFftwTraits::executeC2R(plan, tmp_in.get(), out);
                return;
            }

            std::unique_ptr<double, detail::FftwTypedFree<double>> tmp_out(
                static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * full_real_size)));
            RealFftwTraits::executeC2R(plan, tmp_in.get(), tmp_out.get());
            std::memcpy(out, tmp_out.get(), sizeof(double) * full_real_size);
        }
    } // namespace

    struct RealFFT::Impl {
        RealPlanPtr forward_plan;
        RealPlanPtr inverse_plan;
        std::vector<int> dims;
        int howmany = 1;
        int real_size = 0;
        int complex_size = 0;
        int full_real_size = 0;
        int full_complex_size = 0;
        int alignment = 0;
    };

    RealFFT::RealFFT(std::vector<int> dims, int howmany, PlanRigor rigor)
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
        impl_->howmany = howmany;
        impl_->real_size = real_size;
        impl_->complex_size = complex_size;
        impl_->full_real_size = detail::checkedMultiply(real_size, howmany, "RealFFT real");
        impl_->full_complex_size = detail::checkedMultiply(complex_size, howmany, "RealFFT complex");

        std::unique_ptr<double, detail::FftwTypedFree<double>> buf_in(
            static_cast<double*>(detail::checkedFftwMalloc(sizeof(double) * impl_->full_real_size)));
        std::unique_ptr<std::complex<double>, detail::FftwTypedFree<std::complex<double>>> buf_out(
            static_cast<std::complex<double>*>(
                detail::checkedFftwMalloc(sizeof(std::complex<double>) * impl_->full_complex_size)));
        impl_->alignment = RealFftwTraits::alignmentOf(buf_in.get());

        const unsigned flags = detail::toFFTWFlag(rigor);
        std::lock_guard<std::mutex> lock(detail::fftwPlannerMutex());
        detail::ensureDoubleThreadsInitializedLocked();

        impl_->forward_plan = RealPlanPtr(
            RealFftwTraits::planR2C(
                static_cast<int>(impl_->dims.size()), impl_->dims.data(), howmany,
                buf_in.get(), real_size, buf_out.get(), complex_size, flags));
        impl_->inverse_plan = RealPlanPtr(
            RealFftwTraits::planC2R(
                static_cast<int>(impl_->dims.size()), impl_->dims.data(), howmany,
                buf_out.get(), complex_size, buf_in.get(), real_size, flags));

        if (!impl_->forward_plan || !impl_->inverse_plan)
            throw std::runtime_error("FFTW failed to create real FFT plan.");
    }

    RealFFT::~RealFFT() = default;
    RealFFT::RealFFT(RealFFT&&) noexcept = default;
    RealFFT& RealFFT::operator=(RealFFT&&) noexcept = default;

    int RealFFT::rank() const { return static_cast<int>(impl_->dims.size()); }
    int RealFFT::howmany() const { return impl_->howmany; }
    int RealFFT::realSize() const { return impl_->real_size; }
    int RealFFT::complexSize() const { return impl_->complex_size; }
    int RealFFT::fullRealSize() const { return impl_->full_real_size; }
    int RealFFT::fullComplexSize() const { return impl_->full_complex_size; }
    const std::vector<int>& RealFFT::dims() const { return impl_->dims; }

    void RealFFT::rfft(const Real* in, Complex* out) const {
        execute_rfft_safe(
            impl_->forward_plan.get(), impl_->alignment,
            impl_->full_real_size, impl_->full_complex_size, in, out);
    }

    void RealFFT::irfft(const Complex* in, Real* out, bool normalize) const {
        execute_irfft_safe(
            impl_->inverse_plan.get(), impl_->alignment,
            impl_->full_real_size, impl_->full_complex_size, in, out);
        if (normalize) {
            const double scale = 1.0 / static_cast<double>(impl_->real_size);
            for (int i = 0; i < impl_->full_real_size; ++i)
                out[i] *= scale;
        }
    }

    template <int Rank>
    void RealFFT::rfft(const TensorXr<double, Rank>& in,
                       TensorXc<double, Rank>& out) const {
        rfft(in.data(), out.data());
    }

    template <int Rank>
    void RealFFT::irfft(const TensorXc<double, Rank>& in,
                        TensorXr<double, Rank>& out,
                        bool normalize) const {
        irfft(in.data(), out.data(), normalize);
    }

    template void RealFFT::rfft(const TensorXr<double, 1>&, TensorXc<double, 1>&) const;
    template void RealFFT::rfft(const TensorXr<double, 2>&, TensorXc<double, 2>&) const;
    template void RealFFT::rfft(const TensorXr<double, 3>&, TensorXc<double, 3>&) const;
    template void RealFFT::irfft(const TensorXc<double, 1>&, TensorXr<double, 1>&, bool) const;
    template void RealFFT::irfft(const TensorXc<double, 2>&, TensorXr<double, 2>&, bool) const;
    template void RealFFT::irfft(const TensorXc<double, 3>&, TensorXr<double, 3>&, bool) const;

} // namespace sirius
