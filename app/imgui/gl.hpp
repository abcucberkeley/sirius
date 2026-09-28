#ifndef SIRIUS_IMGUI_GL_HPP
#define SIRIUS_IMGUI_GL_HPP

// OpenGL 3.3 through glad's single-header loader (the one GLFW ships in its
// deps/), and the two small owners the GUI needs: a 2D RGBA texture Dear
// ImGui can draw, and an off-screen colour target the 3D view renders into.
//
// Everything here belongs to the thread that owns the context (the GUI
// thread); the destructors must run while the context is still current.

#include <glad/gl.h>

#include <imgui.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::app::gui {

    // An RGBA8 texture. upload() creates it on first use and re-allocates only
    // when the size changes, so a pane that re-renders while scrubbing reuses
    // its storage.
    class Texture {
    public:
        Texture() = default;
        ~Texture();
        Texture(const Texture&) = delete;
        Texture& operator=(const Texture&) = delete;
        Texture(Texture&& o) noexcept;
        Texture& operator=(Texture&& o) noexcept;

        // `rgba` is width * height * 4 bytes, rows top to bottom.
        void upload(const std::uint8_t* rgba, int width, int height, bool smooth = false);
        // 0xAARRGGBB pixels as the slice renderers produce them (QImage's
        // Format_RGB32 layout): converted on the way in.
        void uploadArgb32(const std::uint32_t* argb, int width, int height, bool smooth = false);
        // One byte per pixel, shown as grey.
        void uploadGray(const std::uint8_t* gray, int width, int height, bool smooth = false);
        void setSmooth(bool smooth);
        void reset();

        bool valid() const noexcept { return id_ != 0; }
        int width() const noexcept { return w_; }
        int height() const noexcept { return h_; }
        GLuint id() const noexcept { return id_; }
        ImTextureRef ref() const noexcept { return ImTextureRef(static_cast<ImTextureID>(id_)); }

    private:
        GLuint id_ = 0;
        int w_ = 0, h_ = 0;
        bool smooth_ = false;
        std::vector<std::uint8_t> scratch_;
    };

    // A colour (+ depth) framebuffer of a given size, re-created when the
    // size changes. The colour attachment is an RGBA8 texture Dear ImGui draws
    // with ImGui::Image(target.ref(), size, ImVec2(0, 1), ImVec2(1, 0)) -- the
    // rows of a framebuffer run bottom to top.
    class RenderTarget {
    public:
        RenderTarget() = default;
        ~RenderTarget();
        RenderTarget(const RenderTarget&) = delete;
        RenderTarget& operator=(const RenderTarget&) = delete;

        // True when the target is complete and bound, with the viewport set.
        bool begin(int width, int height);
        // Restores the framebuffer and viewport that were bound at begin().
        void end();
        void reset();
        // The pixels, rows top to bottom (for a screenshot or a figure).
        std::vector<std::uint8_t> readRgba();

        bool valid() const noexcept { return fbo_ != 0; }
        int width() const noexcept { return w_; }
        int height() const noexcept { return h_; }
        ImTextureRef ref() const noexcept { return ImTextureRef(static_cast<ImTextureID>(color_)); }

    private:
        GLuint fbo_ = 0, color_ = 0, depth_ = 0;
        int w_ = 0, h_ = 0;
        GLint prevFbo_ = 0;
        GLint prevViewport_[4] = {0, 0, 0, 0};
    };

    // Compiles and links a program; 0 (and `log` filled) on failure.
    GLuint buildProgram(const char* vertexSource, const char* fragmentSource, std::string* log = nullptr);

    // PNG in and out (stb). RGBA, rows top to bottom.
    bool writePng(const std::string& path, const std::uint8_t* rgba, int width, int height);
    bool readImage(const std::string& path, std::vector<std::uint8_t>& rgba, int& width, int& height);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_GL_HPP
