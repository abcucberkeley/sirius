#ifndef SIRIUS_FFT_UTIL_HPP
#define SIRIUS_FFT_UTIL_HPP

#include <Eigen/Core>

namespace sirius {

    // Return the DFT sample frequencies for an n-point transform.
    //
    //   f[k] = k / (n * d)           for 0 <= k <= floor((n-1)/2)
    //   f[k] = (k - n) / (n * d)     otherwise
    // 
    // where n is the number of points, and d is the sample spacing. 
    // f has units of inverse d.
    // Example for n = 8, d = 1 
    // k:              0   1   2   3   4   5   6   7
    // integer index:  0   1   2   3  -4  -3  -2  -1
    // f[k]:           0  1/8 2/8 3/8 -4/8 -3/8 -2/8 -1/8
    inline Eigen::VectorXd fftfreq(int n, double d = 1.0) {
        if (n <= 0)
            throw std::invalid_argument("fftfreq: n must be positive");
        if (d == 0.0)
            throw std::invalid_argument("fftfreq: d must be non-zero");

        Eigen::VectorXd f(n);
        const int n_pos = (n - 1) / 2 + 1; // count of non-negative bin indices

        for (int k = 0; k < n_pos; ++k)
            f(k) = static_cast<double>(k);

        for (int k = n_pos; k < n; ++k)
            f(k) = static_cast<double>(k - n);

        f /= static_cast<double>(n) * d;
        return f;
    }

    // In-place version
    inline void fftfreq(int n, double d, Eigen::Ref<Eigen::VectorXd> out) {
        if (n <= 0)
            throw std::invalid_argument("fftfreq: n must be positive");
        if (d == 0.0)
            throw std::invalid_argument("fftfreq: d must be non-zero");
        if (out.size() != n)
            throw std::invalid_argument("fftfreq: out.size() must equal n");

        const int n_pos = (n - 1) / 2 + 1;

        for (int k = 0; k < n_pos; ++k)
            out(k) = static_cast<double>(k);

        for (int k = n_pos; k < n; ++k)
            out(k) = static_cast<double>(k - n);

        out /= static_cast<double>(n) * d;
    }

    // TODO: future utilities (fftshift, ifftshift, rfftfreq, nd grids, ...)

} // namespace sirius

#endif // SIRIUS_FFT_UTIL_HPP