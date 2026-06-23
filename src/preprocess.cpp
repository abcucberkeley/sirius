#include "sirius/preprocess.hpp"
#include "sirius/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sirius {

    void bleach_rescale(Eigen::Tensor<double, 5, Eigen::RowMajor>& data,
                        bool equalizez) {
        using Index = Eigen::Index;

        const Index ndirs   = data.dimension(0);
        const Index nphases = data.dimension(1);
        const Index nz      = data.dimension(2);
        const Index ny      = data.dimension(3);
        const Index nx      = data.dimension(4);
        const Index sec     = ny * nx; // elements per 2D section

        if (data.size() == 0 || nphases == 0) return;
        
        // Get pointer to data
        double* base = data.data();

        // Pointer to the start location of data[d, p, z, :, :]
        auto sectionPtr = [&](Index d, Index p, Index z) -> double* {
            return base + (((d * nphases + p) * nz + z) * sec);
        };

        // Convert 3D section coordinate (d, p, z) to  a flat index
        auto sidx = [&](Index d, Index p, Index z) -> std::size_t {
            return static_cast<std::size_t>((d * nphases + p) * nz + z);
        };
        
        // Flatten (d, z) into one loop: so manual flattening is what actually parallelizes across
        // all ndirs*nz pairs (ndirs alone is ~3 -> most cores idle otherwise).
        const Index ndz = ndirs * nz;
        
        // Plane sums S(d, p, z) over (ny, nx)
        std::vector<double> S(static_cast<std::size_t>(ndirs * nphases * nz));
        #pragma omp parallel for schedule(static)
        for (Index dz = 0; dz < ndz; ++dz) {
            const Index d = dz / nz; // dir index
            const Index z = dz % nz; // z index
            for (Index p = 0; p < nphases; ++p) {
                // Get pointer to the start of data[d, p, z, :, :]
                const double* s = sectionPtr(d, p, z);
                // Sum over (ny, nx)
                double acc = 0.0;
                for (Index i = 0; i < sec; ++i) acc += s[i];
                S[sidx(d, p, z)] = acc;
            }
        }

        // Scale every section to the direction-0/phase-0 reference.
        //   ref = equalizez ? S(0, 0, 0) : S(0, 0, z)
        #pragma omp parallel for schedule(static)
        for (Index dz = 0; dz < ndz; ++dz) {
            const Index d = dz / nz; // dir index
            const Index z = dz % nz; // z index
            const double ref = equalizez ? S[sidx(0, 0, 0)] : S[sidx(0, 0, z)]; 
            for (Index p = 0; p < nphases; ++p) {
                const double s = S[sidx(d, p, z)];
                if (s == 0.0) continue; // dark section: avoid division by zero
                // sums should match reference
                const double f = ref / s; // correction factor
                double* sp = sectionPtr(d, p, z); // Pointer to the start location of data[d, p, z, :, :]
                for (Index i = 0; i < sec; ++i) sp[i] *= f; // scale each image (ny, nx) by the correction factor
            }
        }
    }

    void edge_apodization(Eigen::Tensor<double, 5, Eigen::RowMajor>& data, int napodize) {
        using Index = Eigen::Index;

        const Index ndirs   = data.dimension(0);
        const Index nphases = data.dimension(1);
        const Index nz      = data.dimension(2);
        const Index ny      = data.dimension(3);
        const Index nx      = data.dimension(4);
        const Index sec     = ny * nx; // elements per 2D section

        if (napodize <= 0 || data.size() == 0) return;

        // Border can't exceed the image. fact() still normalizes by the
        // requested napodize, so the taper shape is unchanged for typical inputs.
        const Index nap_y = std::min<Index>(static_cast<Index>(napodize), ny);
        const Index nap_x = std::min<Index>(static_cast<Index>(napodize), nx);

        // Taper weight
        auto fact = [napodize](Index i) -> double {
            return 1.0 - std::sin((static_cast<double>(i) + 0.5) / napodize * kPi * 0.5);
        };

        // NOTE: row stride is nx, NOT (nx/2+1)*2 as in the cudasirecon kernels
        // those run on FFT-padded r2c buffers
        double* base = data.data();
        const Index nsec = ndirs * nphases * nz; // number of sections

        #pragma omp parallel for schedule(static)
        for (Index s = 0; s < nsec; ++s) {
            double* img = base + s * sec; // section pointer

            // Blend top and bottom edges, per column k. diff is read from the
            // original corner rows before either is modified.
            for (Index k = 0; k < nx; ++k) {
                const double diff = (img[(ny - 1) * nx + k] - img[k]) * 0.5;
                for (Index l = 0; l < nap_y; ++l) {
                    const double f = diff * fact(l);
                    img[l * nx + k]            += f;
                    img[(ny - 1 - l) * nx + k] -= f;
                }
            }

            // Blend left and right edges, per row l. This runs after the x-pass
            // (sees its result), matching the sequential ordering of the kernels.
            for (Index l = 0; l < ny; ++l) {
                double* row = img + l * nx;
                const double diff = (row[nx - 1] - row[0]) * 0.5;
                for (Index k = 0; k < nap_x; ++k) {
                    const double f = diff * fact(k);
                    row[k]          += f;
                    row[nx - 1 - k] -= f;
                }
            }
        }
    }

    void cosine_apodization(Eigen::Tensor<double, 5, Eigen::RowMajor>& data) {
        using Index = Eigen::Index;

        const Index ndirs   = data.dimension(0);
        const Index nphases = data.dimension(1);
        const Index nz      = data.dimension(2);
        const Index ny      = data.dimension(3);
        const Index nx      = data.dimension(4);
        const Index sec     = ny * nx;

        if (data.size() == 0) return;

        // The window is separable, so precompute the x and y factors once.
        std::vector<double> xf(static_cast<std::size_t>(nx));
        std::vector<double> yf(static_cast<std::size_t>(ny));
        for (Index k = 0; k < nx; ++k)
            xf[static_cast<std::size_t>(k)] = std::sin(kPi * (static_cast<double>(k) + 0.5) / nx);
        for (Index l = 0; l < ny; ++l)
            yf[static_cast<std::size_t>(l)] = std::sin(kPi * (static_cast<double>(l) + 0.5) / ny);

        double* base = data.data();
        const Index nsec = ndirs * nphases * nz;

        #pragma omp parallel for schedule(static)
        for (Index s = 0; s < nsec; ++s) {
            double* img = base + s * sec;
            for (Index l = 0; l < ny; ++l) {
                const double yfact = yf[static_cast<std::size_t>(l)];
                double* row = img + l * nx;
                for (Index k = 0; k < nx; ++k)
                    row[k] *= xf[static_cast<std::size_t>(k)] * yfact;
            }
        }
    }

} // namespace sirius
