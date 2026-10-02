# Containers

Both images start from the same `nvidia/cuda:<ver>-devel-ubuntu24.04` base and
run the same `install-deps.sh`, so the local Docker environment and the
cluster Apptainer environment are identical. All library dependencies
(Eigen, libtiff and its codecs -- zlib, libdeflate, zstd, libjpeg-turbo --, FFTW,
nvTIFF, nvCOMP, ...) are fetched by CMake, not by the
container, so the images stay small and the dependency pins live in one place
(`cmake/Dependencies.cmake`, `cmake/NvidiaRedist.cmake`).

| file | purpose |
| --- | --- |
| `Dockerfile` | local development: `docker build -f containers/Dockerfile -t sirius:dev .` |
| `sirius.def` | cluster: `apptainer build sirius.sif containers/sirius.def` |
| `install-deps.sh` | shared provisioning (compilers, CMake, Ninja, NASM, OpenMPI, Python) |

Run the C++ tests inside Docker with GPU access:

```bash
docker run --rm -it --gpus all -v "$PWD":/work -w /work sirius:dev bash -c \
  "cmake --preset linux-cuda-dev && cmake --build --preset linux-cuda-dev && ctest --preset linux-cuda-dev"
```

On the cluster (`--nv` exposes the host driver and GPUs):

```bash
apptainer exec --nv sirius.sif cmake --preset linux-cuda-release
apptainer exec --nv sirius.sif cmake --build --preset linux-cuda-release
```

An Apptainer image can also be produced from the Docker image without
rebuilding: `apptainer build sirius.sif docker-daemon://sirius:dev`.

The CUDA toolkit inside the image must not be newer than the host driver
supports (CUDA 13.x needs driver >= 580). Both files take a `CUDA_VERSION`
build argument to select an older line, e.g. `12.8.1`; CMake picks the matching
nvTIFF/nvCOMP redistributable automatically.
