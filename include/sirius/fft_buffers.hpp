#ifndef SIRIUS_FFT_BUFFERS_HPP
#define SIRIUS_FFT_BUFFERS_HPP

#include <Eigen/Core>
#include <stdexcept>
#include <fftw3.h>
#include <limits>

namespace sirius {

    // 1D Complex buffer
    class FFTWBuffer1D {
    public:
        FFTWBuffer1D(Eigen::Index n): n_(n) {
            if (n <= 0) throw std::invalid_argument("Size must be positive");
            if (n > static_cast<Eigen::Index>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            }
            data_ = fftw_alloc_complex(static_cast<size_t>(n));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer1D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer1D(const FFTWBuffer1D&) = delete;
        FFTWBuffer1D& operator=(const FFTWBuffer1D&) = delete;

        // move constructor
        FFTWBuffer1D(FFTWBuffer1D&& other) noexcept: data_(other.data_), n_(other.n_) {
            other.data_ = nullptr;
            other.n_ = 0;
        }
        FFTWBuffer1D& operator=(FFTWBuffer1D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                n_ = other.n_;
                other.data_ = nullptr;
                other.n_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // lightweight eigen conversion
        Eigen::Map<Eigen::VectorXcd> as_eigen() {
            return {reinterpret_cast<std::complex<double>*>(data_), n_};
        }

        Eigen::Index size() const {return n_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index n_ = 0;
    };

    // 2D Complex buffer
    class FFTWBuffer2D {
    public:
        FFTWBuffer2D(Eigen::Index rows, Eigen::Index cols): rows_(rows), cols_(cols), size_(rows * cols) {
            if (rows <= 0 || cols <= 0) throw std::invalid_argument("Size must be positive");
            auto indmax = static_cast<Eigen::Index>(std::numeric_limits<int>::max());
            if (rows > indmax || cols > indmax || size_ > indmax) {
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            }
            data_ = fftw_alloc_complex(static_cast<size_t>(rows * cols));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer2D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer2D(const FFTWBuffer2D&) = delete;
        FFTWBuffer2D& operator=(const FFTWBuffer2D&) = delete;

        // move constructor
        FFTWBuffer2D(FFTWBuffer2D&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_), size_(other.size_) {
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            other.size_ = 0;
        }

        FFTWBuffer2D& operator=(FFTWBuffer2D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                rows_ = other.rows_;
                cols_ = other.cols_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.rows_ = 0;
                other.cols_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // lightweight eigen conversion
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen() {
            return {reinterpret_cast<std::complex<double>*>(data_), rows_, cols_};
        }

        Eigen::Index rows() const {return rows_;}
        Eigen::Index cols() const {return cols_;}
        Eigen::Index size() const {return size_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index rows_ = 0;
        Eigen::Index cols_ = 0;
        Eigen::Index size_ = 0;
    };


}


#endif // SIRIUS_FFT_BUFFERS_HPP