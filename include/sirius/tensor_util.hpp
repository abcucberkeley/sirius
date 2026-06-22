#ifndef SIRIUS_TENSOR_UTIL_HPP
#define SIRIUS_TENSOR_UTIL_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace sirius {
    // Circular shift (roll), shifts may be negative
    // This is the shared primitive behind fftshift / ifftshift
    // in: input tensor
    // shifts: array of shifts for each dimension
    template <typename Scalar, int Rank>
    Eigen::Tensor<Scalar, Rank, Eigen::RowMajor>
    roll(const Eigen::Tensor<Scalar, Rank, Eigen::RowMajor>& in,
         const std::array<int, static_cast<std::size_t>(Rank)>& shifts)
    {
        // Comments will cover a simple 2D example
        // but the code is general for any rank tensor.

        // eg if in[3,4], dims = [3,4]
        const auto& dims = in.dimensions();
        
        // stride = [4,1] since in[row,col] = in[row*4 + col*1]
        // looping over all ranks is then:
        std::array<Eigen::Index, Rank> stride{};
        stride[Rank - 1] = 1; // last dimension always has stride 1
        if constexpr (Rank >= 2) {
            for (int a = Rank - 2; a >= 0; --a) {
                stride[a] = stride[a + 1] * dims[a + 1];
            }
        }

        // Calculate effective positive shifts for each dimension
        std::array<Eigen::Index, Rank> sh{};
        for (int i = 0; i < Rank; ++i) {
            // eg for shift of 1 in dim 0, sh[0] = 1 % 3 = 1
            const Eigen::Index n = dims[i];
            Eigen::Index s = static_cast<Eigen::Index>(shifts[i]) % n;

            // for shift of -1, -1 % 3 = -1, but we want 2, so add n and mod again
            sh[i] = s < 0 ? (s + n) % n : s;
        }

        // Alloc output tensor
        Eigen::Tensor<Scalar, Rank, Eigen::RowMajor> out(dims);
        const Scalar* src = in.data(); // pointer to input data
        Scalar* dst = out.data(); // pointer to output data
        const Eigen::Index total_size = in.size(); // total number of elements

        // Loop over all elements in the input tensor
        for (Eigen::Index idx = 0; idx < total_size; ++idx) {
            // Calculate where the current index maps to in the output tensor after rolling
            // eg strides = [4,1], shifts = [1,2]
            Eigen::Index rem = idx, dest = 0;
            for (int j = 0; j < Rank; ++j) {
                // eg j=0, stride[0] = 4, coord = rem / 4 = row index
                const Eigen::Index coord = rem / stride[j];
                Eigen::Index didx = coord + sh[j]; // apply corresponding shift eg row index  = row + 1
                if (didx >= dims[j]) didx -= dims[j]; // wrap around if needed
                dest += didx * stride[j]; // calculate destination index eg dest += row*4

                // Update to the remainder for the next dimension
                rem -= coord * stride[j]; // eg j=0, coord = row index, rem = col index
            }
            dst[dest] = src[idx];
        }
        return out;
    }
} // namespace sirius

#endif // SIRIUS_TENSOR_UTIL_HPP