#include "sirius/preprocess.hpp"

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

} // namespace sirius
