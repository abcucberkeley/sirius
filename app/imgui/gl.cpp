#include <string>

#define GLAD_GL_IMPLEMENTATION
#include "imgui/gl.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#define STBI_NO_STDIO
#include <stb_image.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace sirius::app::gui {

    // --- Texture ---------------------------------------------------------------

    Texture::~Texture() { reset(); }

    Texture::Texture(Texture&& o) noexcept : id_(o.id_), w_(o.w_), h_(o.h_), smooth_(o.smooth_) {
        o.id_ = 0;
        o.w_ = o.h_ = 0;
    }

    Texture& Texture::operator=(Texture&& o) noexcept {
        if (this != &o) {
            reset();
            id_ = o.id_;
            w_ = o.w_;
            h_ = o.h_;
            smooth_ = o.smooth_;
            o.id_ = 0;
            o.w_ = o.h_ = 0;
        }
        return *this;
    }

    void Texture::reset() {
        if (id_) glDeleteTextures(1, &id_);
        id_ = 0;
        w_ = h_ = 0;
    }

    void Texture::setSmooth(bool smooth) {
        smooth_ = smooth;
        if (!id_) return;
        GLint prev = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev);
        glBindTexture(GL_TEXTURE_2D, id_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev));
    }

    void Texture::upload(const std::uint8_t* rgba, int width, int height, bool smooth) {
        if (!rgba || width <= 0 || height <= 0) return;
        GLint prev = 0, prevAlign = 4, prevRow = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev);
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
        glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRow);
        const bool fresh = id_ == 0;
        if (fresh) glGenTextures(1, &id_);
        glBindTexture(GL_TEXTURE_2D, id_);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        if (fresh || smooth != smooth_) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            smooth_ = smooth;
        }
        if (fresh || width != w_ || height != h_) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
            w_ = width;
            h_ = height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlign);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, prevRow);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev));
    }

    void Texture::uploadArgb32(const std::uint32_t* argb, int width, int height, bool smooth) {
        if (!argb || width <= 0 || height <= 0) return;
        const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        scratch_.resize(n * 4);
        std::uint8_t* out = scratch_.data();
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t p = argb[i];
            out[4 * i + 0] = static_cast<std::uint8_t>((p >> 16) & 0xFF);
            out[4 * i + 1] = static_cast<std::uint8_t>((p >> 8) & 0xFF);
            out[4 * i + 2] = static_cast<std::uint8_t>(p & 0xFF);
            out[4 * i + 3] = 0xFF;
        }
        upload(out, width, height, smooth);
    }

    void Texture::uploadGray(const std::uint8_t* gray, int width, int height, bool smooth) {
        if (!gray || width <= 0 || height <= 0) return;
        const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        scratch_.resize(n * 4);
        std::uint8_t* out = scratch_.data();
        for (std::size_t i = 0; i < n; ++i) {
            out[4 * i + 0] = out[4 * i + 1] = out[4 * i + 2] = gray[i];
            out[4 * i + 3] = 0xFF;
        }
        upload(out, width, height, smooth);
    }

    // --- RenderTarget -------------------------------------------------------------

    RenderTarget::~RenderTarget() { reset(); }

    void RenderTarget::reset() {
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        if (color_) glDeleteTextures(1, &color_);
        if (depth_) glDeleteRenderbuffers(1, &depth_);
        fbo_ = color_ = depth_ = 0;
        w_ = h_ = 0;
    }

    bool RenderTarget::begin(int width, int height) {
        if (width <= 0 || height <= 0) return false;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo_);
        glGetIntegerv(GL_VIEWPORT, prevViewport_);
        if (!fbo_ || width != w_ || height != h_) {
            reset();
            GLint prevTex = 0, prevRb = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
            glGetIntegerv(GL_RENDERBUFFER_BINDING, &prevRb);
            glGenTextures(1, &color_);
            glBindTexture(GL_TEXTURE_2D, color_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glGenRenderbuffers(1, &depth_);
            glBindRenderbuffer(GL_RENDERBUFFER, depth_);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
            glGenFramebuffers(1, &fbo_);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_, 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
            const bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex));
            glBindRenderbuffer(GL_RENDERBUFFER, static_cast<GLuint>(prevRb));
            if (!ok) {
                glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo_));
                reset();
                return false;
            }
            w_ = width;
            h_ = height;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glViewport(0, 0, w_, h_);
        return true;
    }

    void RenderTarget::end() {
        glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo_));
        glViewport(prevViewport_[0], prevViewport_[1], prevViewport_[2], prevViewport_[3]);
    }

    std::vector<std::uint8_t> RenderTarget::readRgba() {
        std::vector<std::uint8_t> out;
        if (!fbo_) return out;
        GLint prev = 0, prevAlign = 4;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev);
        glGetIntegerv(GL_PACK_ALIGNMENT, &prevAlign);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        std::vector<std::uint8_t> flipped(static_cast<std::size_t>(w_) * static_cast<std::size_t>(h_) * 4);
        glReadPixels(0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, flipped.data());
        glPixelStorei(GL_PACK_ALIGNMENT, prevAlign);
        glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev));
        out.resize(flipped.size());
        const std::size_t row = static_cast<std::size_t>(w_) * 4;
        for (int y = 0; y < h_; ++y)
            std::memcpy(out.data() + static_cast<std::size_t>(y) * row,
                        flipped.data() + static_cast<std::size_t>(h_ - 1 - y) * row, row);
        return out;
    }

    // --- shaders ---------------------------------------------------------------

    namespace {
        GLuint compile(GLenum type, const char* source, std::string* log) {
            const GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &source, nullptr);
            glCompileShader(s);
            GLint ok = GL_FALSE;
            glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
            if (ok) return s;
            if (log) {
                GLint n = 0;
                glGetShaderiv(s, GL_INFO_LOG_LENGTH, &n);
                std::string text(static_cast<std::size_t>(n > 0 ? n : 1), '\0');
                glGetShaderInfoLog(s, n, nullptr, text.data());
                *log += (type == GL_VERTEX_SHADER ? "vertex shader: " : "fragment shader: ") + text;
            }
            glDeleteShader(s);
            return 0;
        }
    } // namespace

    GLuint buildProgram(const char* vertexSource, const char* fragmentSource, std::string* log) {
        const GLuint vs = compile(GL_VERTEX_SHADER, vertexSource, log);
        if (!vs) return 0;
        const GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentSource, log);
        if (!fs) {
            glDeleteShader(vs);
            return 0;
        }
        const GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok = GL_FALSE;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (ok) return p;
        if (log) {
            GLint n = 0;
            glGetProgramiv(p, GL_INFO_LOG_LENGTH, &n);
            std::string text(static_cast<std::size_t>(n > 0 ? n : 1), '\0');
            glGetProgramInfoLog(p, n, nullptr, text.data());
            *log += "link: " + text;
        }
        glDeleteProgram(p);
        return 0;
    }

    // --- images -----------------------------------------------------------------

    bool writePng(const std::string& path, const std::uint8_t* rgba, int width, int height) {
        if (!rgba || width <= 0 || height <= 0) return false;
        return stbi_write_png(path.c_str(), width, height, 4, rgba, width * 4) != 0;
    }

    bool readImage(const std::string& path, std::vector<std::uint8_t>& rgba, int& width, int& height) {
        std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
        if (!f) return false;
        const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        int w = 0, h = 0, comp = 0;
        unsigned char* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &comp, 4);
        if (!data) return false;
        rgba.assign(data, data + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
        stbi_image_free(data);
        width = w;
        height = h;
        return true;
    }

} // namespace sirius::app::gui
