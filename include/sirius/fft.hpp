#ifndef SIRIUS_FFT_HPP
#define SIRIUS_FFT_HPP

#include <string>
#include <vector>
#include <memory>
#include <unsupported/Eigen/CXX11/tensor>

namespace sirius {

    // Only support eigen complex double tensors in rowmajor format for now
    template <int Rank>
    using TensorXcd = Eigen::Tensor<std::complex<double>, Rank, Eigen::RowMajor>;

    // Planning rigor controls the time FFTW spends searching for an optimal plan.
    // Higher rigor = better runtime FFT performance, but longer one-time planning cost.
    // Use Estimate for exploratory work; Measure or Patient for production runs on
    // fixed-size transforms that execute many times.
    enum class PlanRigor {
        Estimate,   // No measurement. Fast planning, suboptimal execution.
        Measure,    // Measure a few strategies. Good balance (seconds of planning).
        Patient,    // Measure many strategies. Better plan, slower to create.
        Exhaustive, // Try everything. Rarely worth it over Patient.
    };

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