#include "sirius/fft.hpp"

#include <mutex>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <fftw3.h>

namespace sirius {
    namespace {
        // RAII for fft plan
        // Note: fftw_plan_s is a struct and fftw_plan is a pointer to that struct
        struct FFTWPlanDeleter {
            void operator()(fftw_plan plan) const {fftw_destroy_plan(plan);}
        };
        using PlanPtr = std::unique_ptr<fftw_plan_s, FFTWPlanDeleter>;

        // safe fft buffers
        struct FftwFree { void operator()(void* p) const { fftw_free(p); } };
        using FftwBuf = std::unique_ptr<fftw_complex[], FftwFree>;

        // FFTW's planner modifies global state — must be serialized across all instances
        std::mutex s_planner_mutex;

        // map plan rigor to fftw flags
        unsigned int toFFTWFlag(PlanRigor r) {
            switch (r) {
                case PlanRigor::Estimate:   return FFTW_ESTIMATE;
                case PlanRigor::Measure:    return FFTW_MEASURE;
                case PlanRigor::Patient:    return FFTW_PATIENT;
                case PlanRigor::Exhaustive: return FFTW_EXHAUSTIVE;
            }
            throw std::invalid_argument("Unknown PlanRigor value");
        }

        // safe execution in case unaligned buffers with offset are passed
        void execute_safe(fftw_plan plan, int plan_alignment, int full_size, 
            const std::complex<double>* in, std::complex<double>* out)
        {
            // case to fftw_complex*
            auto* in_ptr  = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in));
            auto* out_ptr = reinterpret_cast<fftw_complex*>(out);

            // check if aligned
            bool aligned = fftw_alignment_of(reinterpret_cast<double*>(in_ptr))  == plan_alignment && 
                           fftw_alignment_of(reinterpret_cast<double*>(out_ptr)) == plan_alignment;
            
            // if aligned, simply execute, otherwise need to copy
            if (aligned) {
                fftw_execute_dft(plan, in_ptr, out_ptr);
            } else {
                FftwBuf tmp_in (static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * full_size)));
                FftwBuf tmp_out(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * full_size)));
                std::memcpy(tmp_in.get(),  in_ptr,  sizeof(fftw_complex) * full_size);
                fftw_execute_dft(plan, tmp_in.get(), tmp_out.get());
                std::memcpy(out_ptr, tmp_out.get(), sizeof(fftw_complex) * full_size);
            }
        }

        // FFTW wisdom is global state no needs shared mutex
        void loadWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            fftw_import_wisdom_from_filename(path.c_str()); // returns 0 on missing file, silently ok
        }

        void saveWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            if (!fftw_export_wisdom_to_filename(path.c_str()))
                throw std::runtime_error("Failed to save FFTW wisdom to: " + path);
        }

    } // anonymous namespace

    struct FFT::Impl {
        PlanPtr forward_plan;
        PlanPtr inverse_plan;
        int total_size = 0; // product of dims
        int full_size = 0; // total_size * howmany
        int alignment = 0;
    };

    FFT::FFT(std::vector<int> dims, int howmany, PlanRigor rigor): impl_(std::make_unique<Impl>())
    {
        if (dims.empty() || dims.size() > 3)
            throw std::invalid_argument("Only ranks 1, 2 and 3 are supported.");

        if (howmany < 1)
            throw std::invalid_argument("howmany must be >= 1");
        
        int total = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>{});
        impl_->total_size = total;
        impl_->full_size = total * howmany;

        FftwBuf buf_in (static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * impl_->full_size)));
        FftwBuf buf_out(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * impl_->full_size)));
        impl_->alignment = fftw_alignment_of(reinterpret_cast<double*>(buf_in.get()));

        unsigned flags = toFFTWFlag(rigor);

        std::lock_guard<std::mutex> lock(s_planner_mutex);

        impl_->forward_plan = PlanPtr(
            fftw_plan_many_dft(
                (int) dims.size(), dims.data(), howmany,
                buf_in.get(), nullptr, 1, total,
                buf_out.get(), nullptr, 1, total,
                FFTW_FORWARD, flags
            )
        );

        impl_->inverse_plan = PlanPtr(
            fftw_plan_many_dft(
                (int) dims.size(), dims.data(), howmany,
                buf_in.get(), nullptr, 1, total,
                buf_out.get(), nullptr, 1, total,
                FFTW_BACKWARD, flags
            )
        );

        if (!impl_->forward_plan || !impl_->inverse_plan) {
            throw std::runtime_error("FFTW failed to create plan.");
        }
    }

    FFT::~FFT() = default;
    FFT::FFT(FFT&&) noexcept = default;
    FFT& FFT::operator=(FFT&&) noexcept = default;

    // Raw interface
    void FFT::fft(const std::complex<double>* in, std::complex<double>* out) const {
        execute_safe(impl_->forward_plan.get(), impl_->alignment, impl_->full_size, in, out);
    }

    void FFT::ifft(const std::complex<double>* in, std::complex<double>* out) const {
        execute_safe(impl_->inverse_plan.get(), impl_->alignment, impl_->full_size, in, out);
    }

    // Convenience functions for eigen
    template<int Rank>
    void FFT::fft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out) const {
        fft(in.data(), out.data());
    }

    template<int Rank>
    void FFT::ifft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out, bool normalize) const {
        ifft(in.data(), out.data());
        if (normalize) out = out / static_cast<double>(impl_->total_size);
    }

    // explicit instantiations to avoid linker errors (templates defined in .cpp must be instantiated there)
    template void FFT::fft(const TensorXcd<1>&, TensorXcd<1>&) const;
    template void FFT::fft(const TensorXcd<2>&, TensorXcd<2>&) const;
    template void FFT::fft(const TensorXcd<3>&, TensorXcd<3>&) const;
    template void FFT::ifft(const TensorXcd<1>&, TensorXcd<1>&, bool) const;
    template void FFT::ifft(const TensorXcd<2>&, TensorXcd<2>&, bool) const;
    template void FFT::ifft(const TensorXcd<3>&, TensorXcd<3>&, bool) const;


    void FFT::loadWisdom(const std::string& path) { loadWisdomImpl(path); }

    void FFT::saveWisdom(const std::string& path) { saveWisdomImpl(path); }

} // namespace sirius