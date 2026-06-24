#ifndef SIRIUS_FFT_HPP
#define SIRIUS_FFT_HPP

#include <memory>
#include <complex>
#include <string>
#include <vector>

#include "sirius/fft_common.hpp"
#include "sirius/tensor_util.hpp"

namespace sirius {

    class FFT {
    public:
        // dims: {n}                 for 1D
        //       {rows, cols}        for 2D
        //       {depth, rows, cols} for 3D
        explicit FFT(std::vector<int> dims, int howmany=1, PlanRigor rigor = PlanRigor::Measure);
        ~FFT();

        // delete copy constructors
        FFT(const FFT&) = delete;
        FFT& operator=(const FFT&) = delete;

        // move constructors
        FFT(FFT&&) noexcept;
        FFT& operator=(FFT&&) noexcept;

        // Raw interface
        void fft(const std::complex<double>* in, std::complex<double>* out) const;
        void ifft(const std::complex<double>* in, std::complex<double>* out) const;

        // Convenience functions for eigen
        template<int Rank>
        void fft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out) const;

        template<int Rank>
        void ifft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out, bool normalize = false) const;

        // Load/Save wisdom from file
        static void loadWisdom(const std::string& path);
        static void saveWisdom(const std::string& path);

    private:
        // Use Pimpl (pointer to implementation) pattern for fftw plan vars
        // otherwise fftw details would have to be exposed to the consumer of the header
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace sirius

#endif // SIRIUS_FFT_HPP