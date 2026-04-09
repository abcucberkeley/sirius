#ifndef SIRIUS_TIFF_IO_HPP
#define SIRIUS_TIFF_IO_HPP

#include <Eigen/Core>
#include <cassert>
#include <string>
#include <memory>

namespace sirius {

    template <typename T>
    using Image = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Compression options for writing
    enum class TiffCompression {
        None,
        Lzw,
        Deflate // Often referred to as ZIP
    };

    template <typename T>
    class ImageStack {
        std::unique_ptr<T[]> data_;
        size_t size_{0};
        Eigen::Index depth_{0}, rows_{0}, cols_{0};
        Eigen::Index stride_{0};

    public:
        ImageStack() noexcept = default;

        // delete copy
        ImageStack(const ImageStack&) = delete;
        ImageStack& operator=(const ImageStack&) = delete;
        
        // default move
        ImageStack(ImageStack&&) noexcept = default;
        ImageStack& operator=(ImageStack&&) noexcept = default;

        ImageStack(Eigen::Index depth, Eigen::Index rows, Eigen::Index cols)
        : data_(std::make_unique<T[]>(
            (assert(depth > 0 && rows > 0 && cols > 0),
            static_cast<size_t>(depth * rows * cols))
        )),
        size_(static_cast<size_t>(depth * rows * cols)),
        depth_(depth), rows_(rows), cols_(cols), stride_(rows * cols) {}

        T& operator()(Eigen::Index z, Eigen::Index r, Eigen::Index c) noexcept {
            assert(z >= 0 && z < depth_ && r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return data_[static_cast<size_t>(z * stride_ + r * cols_ + c)];
        }

        const T& operator()(Eigen::Index z, Eigen::Index r, Eigen::Index c) const noexcept {
            assert(z >= 0 && z < depth_ && r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return data_[static_cast<size_t>(z * stride_ + r * cols_ + c)];
        }

        Eigen::Map<Image<T>> slice(Eigen::Index z) noexcept {
            assert(z >= 0 && z < depth_);
            return {data_.get() + z * stride_, rows_, cols_};
        }

        Eigen::Map<const Image<T>> slice(Eigen::Index z) const noexcept {
            assert(z >= 0 && z < depth_);
            return {data_.get() + z * stride_, rows_, cols_};
        }

        void resize(Eigen::Index depth, Eigen::Index rows, Eigen::Index cols) {
            assert(depth > 0 && rows > 0 && cols > 0);
            depth_ = depth;
            rows_ = rows;
            cols_ = cols;
            stride_ = rows * cols;
            size_ = static_cast<size_t> (depth * stride_);
            data_.reset(new T[size_]);
        }

        T* data() noexcept { return data_.get(); }
        const T* data() const noexcept { return data_.get(); }
        Eigen::Index depth() const noexcept { return depth_; }
        Eigen::Index rows() const noexcept { return rows_; }
        Eigen::Index cols() const noexcept { return cols_; }
        Eigen::Index stride() const noexcept { return stride_; }
        size_t size() const noexcept { return size_; }
        size_t sizeBytes() const noexcept { return size_ * sizeof(T); }
        bool empty() const noexcept { return !data_; }
    };

    template <typename T>
    Image<T> readTiff(const std::string& path);

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path);

    template <typename T>
    void writeTiff(const std::string& path, const Image<T>& image, 
                   TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack, 
                        TiffCompression comp = TiffCompression::None);

} // namespace sirius

#endif // SIRIUS_TIFF_IO_HPP