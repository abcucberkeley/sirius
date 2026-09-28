#include "imgui/viewer/volume_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/constants.hpp>

#include "core/labels.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/viewer/slice_pane.hpp"
#include "imgui/viewer/viewer_constants.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {
        constexpr int kMaxChannels = viewer::kVolumeMaxChannels;

        const char* kVertex = R"(#version 330 core
            out vec2 vNdc;
            void main() {
                // full-screen triangle from gl_VertexID, no buffers needed
                vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
                vNdc = p;
                gl_Position = vec4(p, 0.0, 1.0);
            })";

        const char* kFragment = R"(#version 330 core
            in vec2 vNdc;
            out vec4 fragColor;
            uniform mat4 uInvViewProj;
            uniform vec3 uHalf;            // half extents of the box (world units)
            uniform vec2 uClipZ;           // normalized 0..1 along voxel z
            uniform int uCount;
            uniform sampler3D uTex0, uTex1, uTex2, uTex3;
            uniform vec3 uColor[4];
            uniform float uStep;           // world units per sample
            uniform vec3 uRamp;            // lo, hi, alpha
            uniform int uMip;
            uniform sampler3D uLabels;     // RGBA: palette colour, a = 1 inside a label
            uniform int uLabelsOn;
            uniform float uLabelAlpha;     // label opacity per unit length

            float sampleChannel(int i, vec3 uvw) {
                if (i == 0) return texture(uTex0, uvw).r;
                if (i == 1) return texture(uTex1, uvw).r;
                if (i == 2) return texture(uTex2, uvw).r;
                return texture(uTex3, uvw).r;
            }

            void main() {
                vec4 p0 = uInvViewProj * vec4(vNdc, -1.0, 1.0);
                vec4 p1 = uInvViewProj * vec4(vNdc, 1.0, 1.0);
                vec3 o = p0.xyz / p0.w;
                vec3 d = normalize(p1.xyz / p1.w - o);
                // voxel z runs from the +z face (plane 0) towards -z: clip in that frame
                vec3 lo = -uHalf, hi = uHalf;
                hi.z = uHalf.z - uClipZ.x * 2.0 * uHalf.z;
                lo.z = uHalf.z - uClipZ.y * 2.0 * uHalf.z;
                vec3 inv = 1.0 / d;
                vec3 t0 = (lo - o) * inv, t1 = (hi - o) * inv;
                vec3 tmin = min(t0, t1), tmax = max(t0, t1);
                float tn = max(max(tmin.x, tmin.y), tmin.z);
                float tf = min(min(tmax.x, tmax.y), tmax.z);
                if (tf <= max(tn, 0.0)) { fragColor = vec4(0.0); return; }
                tn = max(tn, 0.0);
                vec3 acc = vec3(0.0);
                float alpha = 0.0;
                vec3 lab = vec3(0.0);          // labels composited front to back on their own
                float labA = 0.0;
                float best[4];
                best[0] = 0.0; best[1] = 0.0; best[2] = 0.0; best[3] = 0.0;
                int steps = int((tf - tn) / uStep) + 1;
                steps = min(steps, 2048);
                for (int s = 0; s < steps; ++s) {
                    float t = tn + float(s) * uStep;
                    if (t > tf) break;
                    vec3 p = o + d * t;
                    // world -> texture: x right, y down (rows), z plane 0 at +z
                    vec3 uvw = vec3((p.x + uHalf.x) / (2.0 * uHalf.x),
                                    (uHalf.y - p.y) / (2.0 * uHalf.y),
                                    (uHalf.z - p.z) / (2.0 * uHalf.z));
                    for (int c = 0; c < uCount; ++c) {
                        float v = sampleChannel(c, uvw);
                        if (uMip == 1) {
                            best[c] = max(best[c], v);
                        } else {
                            float a = clamp((v - uRamp.x) / max(uRamp.y - uRamp.x, 1e-4), 0.0, 1.0) * uRamp.z;
                            a *= uStep * 40.0;      // opacity per unit length, independent of the step
                            a = clamp(a, 0.0, 1.0);
                            acc += (1.0 - alpha) * a * uColor[c] * v;
                            alpha += (1.0 - alpha) * a;
                        }
                    }
                    if (uLabelsOn == 1 && labA < 0.985) {
                        vec4 l = texture(uLabels, uvw);
                        if (l.a > 0.5) {
                            float a = clamp(uLabelAlpha * uStep * 60.0, 0.0, 1.0);
                            lab += (1.0 - labA) * a * l.rgb;
                            labA += (1.0 - labA) * a;
                        }
                    }
                    if (alpha > 0.985 && (uLabelsOn == 0 || labA > 0.985)) break;
                }
                if (uMip == 1) {
                    for (int c = 0; c < uCount; ++c) acc += uColor[c] * best[c];
                    alpha = clamp(max(max(best[0], best[1]), max(best[2], best[3])), 0.0, 1.0);
                }
                // labels in front of the intensity they cover
                fragColor = vec4(lab + (1.0 - labA) * acc, labA + (1.0 - labA) * alpha);
            })";

        // --- 4 x 4 matrices, column-major (what glUniformMatrix4fv takes) ---------
        struct Mat4 {
            float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            float& at(int row, int col) { return m[col * 4 + row]; }
            float at(int row, int col) const { return m[col * 4 + row]; }
        };

        Mat4 multiply(const Mat4& a, const Mat4& b) {
            Mat4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    float s = 0.0f;
                    for (int k = 0; k < 4; ++k) s += a.at(i, k) * b.at(k, j);
                    r.at(i, j) = s;
                }
            return r;
        }

        // The OpenGL perspective projection (gluPerspective's matrix).
        Mat4 perspective(float fovYDeg, float aspect, float nearPlane, float farPlane) {
            Mat4 r;
            const float half = fovYDeg / 2.0f * static_cast<float>(sirius::kPi) / 180.0f;
            const float cotan = std::cos(half) / std::sin(half);
            const float clip = farPlane - nearPlane;
            r.at(0, 0) = cotan / aspect;
            r.at(1, 1) = cotan;
            r.at(2, 2) = -(nearPlane + farPlane) / clip;
            r.at(2, 3) = -(2.0f * nearPlane * farPlane) / clip;
            r.at(3, 2) = -1.0f;
            r.at(3, 3) = 0.0f;
            return r;
        }

        struct V3 {
            float x, y, z;
        };
        V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
        V3 normalized(V3 a) {
            const float n = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
            return n > 0.0f ? V3{a.x / n, a.y / n, a.z / n} : a;
        }
        float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

        // The view matrix from `eye` towards `centre` (gluLookAt's).
        Mat4 lookAt(V3 eye, V3 centre, V3 up) {
            const V3 f = normalized(sub(centre, eye));
            const V3 s = normalized(cross(f, up));
            const V3 u = cross(s, f);
            Mat4 r;
            r.at(0, 0) = s.x, r.at(0, 1) = s.y, r.at(0, 2) = s.z, r.at(0, 3) = -dot(s, eye);
            r.at(1, 0) = u.x, r.at(1, 1) = u.y, r.at(1, 2) = u.z, r.at(1, 3) = -dot(u, eye);
            r.at(2, 0) = -f.x, r.at(2, 1) = -f.y, r.at(2, 2) = -f.z, r.at(2, 3) = dot(f, eye);
            return r;
        }

        Mat4 inverted(const Mat4& a) {
            const float* m = a.m;
            float inv[16];
            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
            float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
            Mat4 r;
            if (det == 0.0f) return r;
            det = 1.0f / det;
            for (int i = 0; i < 16; ++i) r.m[i] = inv[i] * det;
            return r;
        }

        struct Clip4 {
            float x, y, z, w;
        };
        Clip4 transform(const Mat4& a, V3 p) {
            return {a.at(0, 0) * p.x + a.at(0, 1) * p.y + a.at(0, 2) * p.z + a.at(0, 3),
                    a.at(1, 0) * p.x + a.at(1, 1) * p.y + a.at(1, 2) * p.z + a.at(1, 3),
                    a.at(2, 0) * p.x + a.at(2, 1) * p.y + a.at(2, 2) * p.z + a.at(2, 3),
                    a.at(3, 0) * p.x + a.at(3, 1) * p.y + a.at(3, 2) * p.z + a.at(3, 3)};
        }

        // The GL state the rendering touches, put back afterwards: the frame
        // around it is Dear ImGui's, drawn later with the state it expects.
        struct GlStateGuard {
            GLint program = 0, vao = 0, arrayBuffer = 0, activeTexture = 0, tex2d = 0, unpackAlign = 4, unpackRow = 0;
            GLint tex3d[kMaxChannels + 1] = {};
            GLboolean blend = GL_FALSE, depth = GL_FALSE, scissor = GL_FALSE, cull = GL_FALSE;
            GLint srcRgb = 0, dstRgb = 0, srcA = 0, dstA = 0, eqRgb = 0, eqA = 0;
            GLfloat clear[4] = {0, 0, 0, 0};
            GlStateGuard() {
                glGetIntegerv(GL_CURRENT_PROGRAM, &program);
                glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
                glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
                glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
                for (int i = 0; i <= kMaxChannels; ++i) {
                    glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
                    glGetIntegerv(GL_TEXTURE_BINDING_3D, &tex3d[i]);
                }
                glActiveTexture(GL_TEXTURE0);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex2d);
                glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpackAlign);
                glGetIntegerv(GL_UNPACK_ROW_LENGTH, &unpackRow);
                blend = glIsEnabled(GL_BLEND);
                depth = glIsEnabled(GL_DEPTH_TEST);
                scissor = glIsEnabled(GL_SCISSOR_TEST);
                cull = glIsEnabled(GL_CULL_FACE);
                glGetIntegerv(GL_BLEND_SRC_RGB, &srcRgb);
                glGetIntegerv(GL_BLEND_DST_RGB, &dstRgb);
                glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
                glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);
                glGetIntegerv(GL_BLEND_EQUATION_RGB, &eqRgb);
                glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eqA);
                glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
            }
            ~GlStateGuard() {
                glUseProgram(static_cast<GLuint>(program));
                glBindVertexArray(static_cast<GLuint>(vao));
                glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(arrayBuffer));
                for (int i = 0; i <= kMaxChannels; ++i) {
                    glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
                    glBindTexture(GL_TEXTURE_3D, static_cast<GLuint>(tex3d[i]));
                }
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(tex2d));
                glActiveTexture(static_cast<GLenum>(activeTexture));
                glPixelStorei(GL_UNPACK_ALIGNMENT, unpackAlign);
                glPixelStorei(GL_UNPACK_ROW_LENGTH, unpackRow);
                auto set = [](GLenum cap, GLboolean on) {
                    if (on) glEnable(cap);
                    else glDisable(cap);
                };
                set(GL_BLEND, blend);
                set(GL_DEPTH_TEST, depth);
                set(GL_SCISSOR_TEST, scissor);
                set(GL_CULL_FACE, cull);
                glBlendEquationSeparate(static_cast<GLenum>(eqRgb), static_cast<GLenum>(eqA));
                glBlendFuncSeparate(static_cast<GLenum>(srcRgb), static_cast<GLenum>(dstRgb), static_cast<GLenum>(srcA), static_cast<GLenum>(dstA));
                glClearColor(clear[0], clear[1], clear[2], clear[3]);
            }
            GlStateGuard(const GlStateGuard&) = delete;
            GlStateGuard& operator=(const GlStateGuard&) = delete;
        };

        void textureParams(GLenum filter) {
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(filter));
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(filter));
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        }
    } // namespace

    struct VolumeView::Gl {
        GLuint ray = 0;
        GLuint vao = 0;
        GLuint textures[kMaxChannels] = {0, 0, 0, 0};
        int textureCount = 0;
        GLuint labelTexture = 0;
        bool labelsUploaded = false;
        RenderTarget target;
        // physical box and camera of the last rendering (the box lines use them)
        float half[3] = {0, 0, 0};
        Mat4 viewProj;
    };

    VolumeView::VolumeView() : gl_(std::make_unique<Gl>()) {}

    VolumeView::~VolumeView() {
        if (glOk_) {
            glDeleteTextures(kMaxChannels, gl_->textures);
            if (gl_->labelTexture) glDeleteTextures(1, &gl_->labelTexture);
            if (gl_->vao) glDeleteVertexArrays(1, &gl_->vao);
            if (gl_->ray) glDeleteProgram(gl_->ray);
        }
        gl_->target.reset();
    }

    // --- state -----------------------------------------------------------------------

    void VolumeView::setTextures(std::uint64_t key, std::vector<ReducedVolume> channels, const std::array<double, 3>& voxelUm, Index nz,
                                 Index ny, Index nx) {
        textures_ = std::move(channels);
        if (textures_.size() > static_cast<std::size_t>(kMaxChannels)) textures_.resize(kMaxChannels);
        key_ = key;
        voxelUm_ = voxelUm;
        vz_ = nz;
        vy_ = ny;
        vx_ = nx;
        preparing_.clear();
    }

    void VolumeView::clearVolumes() {
        textures_.clear();
        vz_ = vy_ = vx_ = 0;
        key_ = 0;
    }

    void VolumeView::setLabels(std::uint64_t key, std::shared_ptr<const void> owner, const std::uint32_t* labels, Index z, Index y, Index x,
                               float opacity, std::uint32_t only) {
        labels_ = labels;
        labelsOwner_ = std::move(owner);
        labelOnly_ = only;
        lz_ = z;
        ly_ = y;
        lx_ = x;
        labelsKey_ = key;
        labelOpacity_ = opacity;
    }

    void VolumeView::clearLabels() {
        if (!labels_ && labelsKey_ == 0) return;
        labels_ = nullptr;
        labelsOwner_.reset();
        labelsKey_ = 0;
    }

    void VolumeView::applyOrientation(double yaw, double pitch, bool emitSignal) {
        yaw = std::fmod(yaw, 360.0);
        if (yaw < 0) yaw += 360.0;
        pitch = std::clamp(pitch, -60.0, 60.0);
        if (yaw == yaw_ && pitch == pitch_) return;
        yaw_ = yaw;
        pitch_ = pitch;
        if (emitSignal && orientationChanged) orientationChanged(yaw_, pitch_);
    }

    void VolumeView::setClip(double lo, double hi) {
        clipLo_ = std::clamp(lo, 0.0, 1.0);
        clipHi_ = std::clamp(hi, clipLo_, 1.0);
    }

    void VolumeView::setZoom(double zoom) { zoom_ = std::clamp(zoom, viewer::kMinVolumeZoom, viewer::kMaxVolumeZoom); }

    void VolumeView::setTransfer(float lo, float hi, float alpha, float stepVoxels, bool mip) {
        tfLo_ = lo;
        tfHi_ = std::max(hi, lo + 1e-3f);
        tfAlpha_ = std::clamp(alpha, 0.0f, 1.0f);
        stepVoxels_ = std::clamp(stepVoxels, 0.1f, 4.0f);
        mip_ = mip;
    }

    bool VolumeView::grabImage(std::vector<std::uint8_t>& rgba, int& width, int& height) {
        if (!gl_->target.valid()) return false;
        rgba = gl_->target.readRgba();
        width = gl_->target.width();
        height = gl_->target.height();
        for (std::size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
        return !rgba.empty();
    }

    // --- GL ------------------------------------------------------------------------------

    bool VolumeView::initGl() {
        if (glTried_) return glOk_;
        glTried_ = true;
        GLint major = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        if (major < 3) {
            GLint minor = 0;
            glGetIntegerv(GL_MINOR_VERSION, &minor);
            glError_ = format("OpenGL 3.0 or newer is required for volume rendering (got %d.%d)", major, minor);
            return false;
        }
        gl_->ray = buildProgram(kVertex, kFragment, &glError_);
        if (!gl_->ray) return false;
        glGenVertexArrays(1, &gl_->vao);
        glGenTextures(kMaxChannels, gl_->textures);
        glGenTextures(1, &gl_->labelTexture);
        glOk_ = true;
        uploadedKey_ = 0;
        uploadedLabelsKey_ = 0;
        return true;
    }

    // The label volume at the same reduction as the intensity textures,
    // nearest label per box centre, as palette colours with a = 1 where labelled.
    void VolumeView::uploadLabels() {
        gl_->labelsUploaded = false;
        uploadedLabelsKey_ = labelsKey_;
        if (!labels_ || lx_ <= 0 || ly_ <= 0 || lz_ <= 0) return;
        const Index cap = viewer::kVolumeTexelsMax;
        const int fx = static_cast<int>((lx_ + cap - 1) / cap);
        const int fy = static_cast<int>((ly_ + cap - 1) / cap);
        const int fz = static_cast<int>((lz_ + cap - 1) / cap);
        const int tx = static_cast<int>((lx_ + fx - 1) / fx), ty = static_cast<int>((ly_ + fy - 1) / fy),
                  tz = static_cast<int>((lz_ + fz - 1) / fz);
        std::vector<unsigned char> texels(static_cast<std::size_t>(tx) * static_cast<std::size_t>(ty) * static_cast<std::size_t>(tz) * 4, 0);
        bool any = false;
        for (int z = 0; z < tz; ++z) {
            const Index zz = std::min<Index>(static_cast<Index>(z) * fz + fz / 2, lz_ - 1);
            for (int y = 0; y < ty; ++y) {
                const Index yy = std::min<Index>(static_cast<Index>(y) * fy + fy / 2, ly_ - 1);
                const std::uint32_t* row = labels_ + (zz * ly_ + yy) * lx_;
                unsigned char* out = texels.data() + (static_cast<std::size_t>(z) * static_cast<std::size_t>(ty) + static_cast<std::size_t>(y)) *
                                                         static_cast<std::size_t>(tx) * 4;
                for (int x = 0; x < tx; ++x) {
                    const Index xx = std::min<Index>(static_cast<Index>(x) * fx + fx / 2, lx_ - 1);
                    const std::uint32_t id = row[xx];
                    if (!id || (labelOnly_ != 0 && id != labelOnly_)) continue;
                    const std::array<float, 3> col = labelColor(id);
                    const std::size_t o = static_cast<std::size_t>(x) * 4;
                    out[o + 0] = static_cast<unsigned char>(col[0] * 255.0f);
                    out[o + 1] = static_cast<unsigned char>(col[1] * 255.0f);
                    out[o + 2] = static_cast<unsigned char>(col[2] * 255.0f);
                    out[o + 3] = 255;
                    any = true;
                }
            }
        }
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, gl_->labelTexture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        textureParams(GL_NEAREST);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, tx, ty, tz, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels.data());
        glBindTexture(GL_TEXTURE_3D, 0);
        gl_->labelsUploaded = any;
    }

    void VolumeView::uploadTextures() {
        // The reduction happened on the loader thread: this is the upload.
        gl_->textureCount = 0;
        glActiveTexture(GL_TEXTURE0);
        for (const ReducedVolume& brick : textures_) {
            if (brick.texels.empty() || brick.tx <= 0 || brick.ty <= 0 || brick.tz <= 0) continue;
            glBindTexture(GL_TEXTURE_3D, gl_->textures[gl_->textureCount]);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            textureParams(GL_LINEAR);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, brick.tx, brick.ty, brick.tz, 0, GL_RED, GL_UNSIGNED_BYTE, brick.texels.data());
            ++gl_->textureCount;
            if (gl_->textureCount >= kMaxChannels) break;
        }
        glBindTexture(GL_TEXTURE_3D, 0);
        uploadedKey_ = key_;
    }

    void VolumeView::render(int width, int height) {
        // physical box: the longest side is 1 world unit
        std::array<double, 3> ext{0.0, 0.0, 0.0};
        int nz = 1;
        if (!textures_.empty() && vx_ > 0 && vy_ > 0 && vz_ > 0) {
            ext = {static_cast<double>(vx_) * voxelUm_[0], static_cast<double>(vy_) * voxelUm_[1], static_cast<double>(vz_) * voxelUm_[2]};
            nz = static_cast<int>(vz_);
        }
        const double longest = std::max({ext[0], ext[1], ext[2], 1e-9});
        for (int i = 0; i < 3; ++i) gl_->half[i] = static_cast<float>(ext[static_cast<std::size_t>(i)] / longest / 2);

        const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
        const Mat4 proj = perspective(32.0f, aspect, 0.05f, 20.0f);
        const double yaw = yaw_ * sirius::kPi / 180.0, pitch = pitch_ * sirius::kPi / 180.0;
        const double dist = 2.4 / zoom_;
        const V3 cam{static_cast<float>(dist * std::sin(yaw) * std::cos(pitch)), static_cast<float>(dist * std::sin(pitch)),
                     static_cast<float>(dist * std::cos(yaw) * std::cos(pitch))};
        gl_->viewProj = multiply(proj, lookAt(cam, V3{0, 0, 0}, V3{0, 1, 0}));

        const GlStateGuard guard;
        if (!gl_->target.begin(width, height)) return;
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glClearColor(0x0a / 255.0f, 0x09 / 255.0f, 0x09 / 255.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        if (glOk_ && !textures_.empty()) {
            if (uploadedKey_ != key_) uploadTextures();
            if (uploadedLabelsKey_ != labelsKey_) uploadLabels();
            if (gl_->textureCount > 0) {
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                const GLuint p = gl_->ray;
                glUseProgram(p);
                const Mat4 inv = inverted(gl_->viewProj);
                glUniformMatrix4fv(glGetUniformLocation(p, "uInvViewProj"), 1, GL_FALSE, inv.m);
                glUniform3f(glGetUniformLocation(p, "uHalf"), gl_->half[0], gl_->half[1], gl_->half[2]);
                glUniform2f(glGetUniformLocation(p, "uClipZ"), static_cast<float>(clipLo_), static_cast<float>(clipHi_));
                glUniform1i(glGetUniformLocation(p, "uCount"), gl_->textureCount);
                const float stepWorld = stepVoxels_ * (2.0f * gl_->half[2] / static_cast<float>(std::max(nz, 1)));
                glUniform1f(glGetUniformLocation(p, "uStep"), std::max(stepWorld, 0.002f));
                glUniform3f(glGetUniformLocation(p, "uRamp"), tfLo_, tfHi_, tfAlpha_);
                glUniform1i(glGetUniformLocation(p, "uMip"), mip_ ? 1 : 0);
                const char* texNames[kMaxChannels] = {"uTex0", "uTex1", "uTex2", "uTex3"};
                const char* colorNames[kMaxChannels] = {"uColor[0]", "uColor[1]", "uColor[2]", "uColor[3]"};
                int k = 0;
                for (std::size_t i = 0; i < textures_.size() && k < gl_->textureCount; ++i) {
                    const ReducedVolume& brick = textures_[i];
                    if (brick.texels.empty()) continue;
                    glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(k));
                    glBindTexture(GL_TEXTURE_3D, gl_->textures[k]);
                    glUniform1i(glGetUniformLocation(p, texNames[k]), k);
                    glUniform3f(glGetUniformLocation(p, colorNames[k]), brick.color[0], brick.color[1], brick.color[2]);
                    ++k;
                }
                // every sampler needs a unit of its own type, bound or not
                for (int j = k; j < kMaxChannels; ++j) glUniform1i(glGetUniformLocation(p, texNames[j]), j);
                const bool labelsOn = gl_->labelsUploaded && labels_ != nullptr;
                glUniform1i(glGetUniformLocation(p, "uLabelsOn"), labelsOn ? 1 : 0);
                glUniform1f(glGetUniformLocation(p, "uLabelAlpha"), labelOpacity_);
                glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(kMaxChannels));
                glBindTexture(GL_TEXTURE_3D, labelsOn ? gl_->labelTexture : 0);
                glUniform1i(glGetUniformLocation(p, "uLabels"), kMaxChannels);
                glBindVertexArray(gl_->vao);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glActiveTexture(GL_TEXTURE0);
            }
        }
        gl_->target.end();
    }

    // --- drawing ---------------------------------------------------------------------------

    void VolumeView::draw(ImVec2 min, ImVec2 max) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiIO& io = ImGui::GetIO();
        const float w = max.x - min.x, h = max.y - min.y;
        if (w < 1.0f || h < 1.0f) return;

        // input first: the rendering below already follows this frame's drag
        ImGui::SetCursorScreenPos(min);
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##volume", ImVec2(w, h));
        if (ImGui::IsItemActivated()) dragging_ = true;
        if (dragging_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Drag the volume, not the camera: dragging right turns the volume to
            // the right and dragging down tips its top towards the viewer. The
            // camera orbits at (sin yaw, sin pitch, cos yaw), so both angles move
            // against the drag to make the volume follow the pointer. (Degrees
            // per design pixel, so the display scale does not change the speed.)
            const double s = 1.0 / std::max(0.01, static_cast<double>(theme::scale()));
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)
                applyOrientation(yaw_ - static_cast<double>(io.MouseDelta.x) * 0.5 * s, pitch_ + static_cast<double>(io.MouseDelta.y) * 0.5 * s, true);
        } else {
            dragging_ = false;
        }
        if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) {
            setZoom(zoom_ * std::pow(viewer::kWheelZoomBase, static_cast<double>(io.MouseWheel)));
            if (zoomChanged) zoomChanged(zoom_);
        }

        if (initGl()) {
            const int fw = std::max(1, static_cast<int>(std::lround(w * io.DisplayFramebufferScale.x)));
            const int fh = std::max(1, static_cast<int>(std::lround(h * io.DisplayFramebufferScale.y)));
            const std::string state =
                format("%llu %llu %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %d %zu %.3f", static_cast<unsigned long long>(key_),
                       static_cast<unsigned long long>(labelsKey_), fw, fh, yaw_, pitch_, zoom_, clipLo_, clipHi_, static_cast<double>(tfLo_),
                       static_cast<double>(tfHi_), static_cast<double>(tfAlpha_), static_cast<double>(stepVoxels_), mip_ ? 1 : 0,
                       textures_.size(), static_cast<double>(labelOpacity_));
            if (state != renderedState_ || !gl_->target.valid()) {
                render(fw, fh);
                renderedState_ = state;
            }
        }
        if (glOk_ && gl_->target.valid()) dl->AddImage(gl_->target.ref(), min, max, ImVec2(0, 1), ImVec2(1, 0));
        else dl->AddRectFilled(min, max, theme::kViewerGround);

        dl->PushClipRect(min, max, true);
        if (glOk_ && box_ && !textures_.empty()) {
            // 12 edges; the three from the voxel origin (-x, +y, +z corner) in accent
            const float hx = gl_->half[0], hy = gl_->half[1], hz = gl_->half[2];
            const V3 o{-hx, hy, hz};
            struct Edge {
                V3 a, b;
                bool accent;
            };
            const Edge edges[] = {{o, {hx, hy, hz}, true},
                                  {o, {-hx, -hy, hz}, true},
                                  {o, {-hx, hy, -hz}, true},
                                  {{hx, hy, hz}, {hx, -hy, hz}, false},
                                  {{hx, hy, hz}, {hx, hy, -hz}, false},
                                  {{-hx, -hy, hz}, {hx, -hy, hz}, false},
                                  {{-hx, -hy, hz}, {-hx, -hy, -hz}, false},
                                  {{-hx, hy, -hz}, {hx, hy, -hz}, false},
                                  {{-hx, hy, -hz}, {-hx, -hy, -hz}, false},
                                  {{hx, -hy, -hz}, {hx, hy, -hz}, false},
                                  {{hx, -hy, -hz}, {-hx, -hy, -hz}, false},
                                  {{hx, -hy, -hz}, {hx, -hy, hz}, false}};
            auto toScreen = [&](const Clip4& c) {
                return ImVec2(min.x + (c.x / c.w * 0.5f + 0.5f) * w, min.y + (0.5f - c.y / c.w * 0.5f) * h);
            };
            for (int pass = 0; pass < 2; ++pass) {
                for (const Edge& e : edges) {
                    if (e.accent != (pass == 1)) continue;
                    Clip4 a = transform(gl_->viewProj, e.a), b = transform(gl_->viewProj, e.b);
                    // keep the part in front of the near plane (z + w >= 0)
                    const float da = a.z + a.w, db = b.z + b.w;
                    if (da < 0.0f && db < 0.0f) continue;
                    if (da < 0.0f || db < 0.0f) {
                        const float t = da / (da - db);
                        const Clip4 c{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
                        if (da < 0.0f) a = c;
                        else b = c;
                    }
                    if (a.w <= 1e-6f || b.w <= 1e-6f) continue;
                    if (pass == 1) dl->AddLine(toScreen(a), toScreen(b), theme::kAccent, px(1.5f));
                    else dl->AddLine(toScreen(a), toScreen(b), theme::rgb(242, 242, 242, 89), px(1.0f));
                }
            }
        }

        // overlays on top of the rendering
        const ImVec2 at(min.x + px(10), min.y + px(8));
        float x = at.x + drawOverlayText(dl, at, "VOLUME", true) + px(12);
        x += drawOverlayText(dl, ImVec2(x, at.y), method_, false, 0.7f) + px(12);
        drawOverlayText(dl, ImVec2(x, at.y),
                        format("yaw %ld\xC2\xB0 \xC2\xB7 pitch %ld\xC2\xB0", std::lround(yaw_), std::lround(pitch_)), false, 0.7f);
        if (!glOk_) {
            drawCenteredWrapped(dl, ImVec2(min.x + px(12), min.y + px(12)), ImVec2(max.x - px(12), max.y - px(12)),
                                "Volume rendering unavailable: " + glError_, 12, theme::rgb(243, 242, 242, 180));
        } else if (textures_.empty()) {
            drawCenteredWrapped(dl, min, max, preparing_.empty() ? std::string("No volume to render") : preparing_, 12,
                                theme::rgb(243, 242, 242, preparing_.empty() ? 140 : 200));
        } else if (!preparing_.empty()) {
            drawOverlayText(dl, ImVec2(min.x + px(static_cast<float>(viewer::kOverlayInset)), min.y + px(static_cast<float>(viewer::kOverlayTop) + 18)),
                            preparing_, false, 0.75f);
        }
        dl->PopClipRect();
        drawControls(min, max);
    }

    // Presets bottom-left, yaw / pitch sliders bottom-right, the Z clip top-right.
    void VolumeView::drawControls(ImVec2 min, ImVec2 max) {
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);

        // presets
        static const struct {
            const char* name;
            double yaw, pitch;
        } presets[] = {{"Front", 0, 0}, {"Iso", 35, 22}, {"Top", 0, 60}, {"Side", 90, 0}};
        const ImVec2 presetSize(theme::snap(px(4 * 52 + 3 * 8)), theme::snap(px(22)));
        ImGui::SetCursorScreenPos(ImVec2(min.x + px(10), max.y - px(10) - presetSize.y));
        if (ImGui::BeginChild("##volPresets", presetSize, ImGuiChildFlags_None, flags)) {
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            for (int i = 0; i < 4; ++i) {
                ImGui::SetCursorScreenPos(ImVec2(origin.x + static_cast<float>(i) * px(60), origin.y));
                widgets::GlyphOpts o;
                o.onDark = true;
                o.glyphPx = 11;
                o.active = std::abs(presets[i].yaw - yaw_) < 0.5 && std::abs(presets[i].pitch - pitch_) < 0.5;
                ImGui::PushID(i);
                if (widgets::glyphTextButton("##preset", presets[i].name, ImVec2(52, 22), o)) applyOrientation(presets[i].yaw, presets[i].pitch, true);
                ImGui::PopID();
            }
        }
        ImGui::EndChild();

        // yaw / pitch
        const float labelW = std::max(theme::textSize("Yaw", 11).x, theme::textSize("Pitch", 11).x);
        const ImVec2 sliderSize(theme::snap(labelW + px(10) + px(140)), theme::snap(px(18 + 6 + 18)));
        ImGui::SetCursorScreenPos(ImVec2(max.x - px(10) - sliderSize.x, max.y - px(10) - sliderSize.y));
        if (ImGui::BeginChild("##volSliders", sliderSize, ImGuiChildFlags_None, flags)) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            widgets::SliderOpts so;
            so.width = 140;
            so.onDark = true;
            auto row = [&](float y, const char* label, const char* id, std::int64_t value, std::int64_t lo, std::int64_t hi, bool isYaw) {
                const ImVec2 ts = theme::textSize(label, 11);
                widgets::drawText(dl, ImVec2(origin.x, origin.y + y + (px(18) - ts.y) * 0.5f), label, 11, theme::kViewerText);
                ImGui::SetCursorScreenPos(ImVec2(origin.x + labelW + px(10), origin.y + y));
                std::int64_t v = value;
                if (widgets::sliderInt(id, &v, lo, hi, so)) {
                    if (isYaw) applyOrientation(static_cast<double>(v), pitch_, true);
                    else applyOrientation(yaw_, static_cast<double>(v), true);
                }
            };
            row(0.0f, "Yaw", "##yaw", std::lround(yaw_) % 360, 0, 359, true);
            row(px(24), "Pitch", "##pitch", std::lround(pitch_), -60, 60, false);
        }
        ImGui::EndChild();

        // Z clip
        const float clipLabelW = theme::textSize("Clip Z", 11).x;
        const ImVec2 clipSize(theme::snap(clipLabelW + px(8) + px(120)), theme::snap(px(18)));
        ImGui::SetCursorScreenPos(ImVec2(max.x - px(10) - clipSize.x, min.y + px(8)));
        if (ImGui::BeginChild("##volClip", clipSize, ImGuiChildFlags_None, flags)) {
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 ts = theme::textSize("Clip Z", 11);
            widgets::drawText(ImGui::GetWindowDrawList(), ImVec2(origin.x, origin.y + (px(18) - ts.y) * 0.5f), "Clip Z", 11, theme::kViewerText);
            ImGui::SetCursorScreenPos(ImVec2(origin.x + clipLabelW + px(8), origin.y));
            widgets::SliderOpts so;
            so.width = 120;
            so.onDark = true;
            double lo = clipLo_, hi = clipHi_;
            if (widgets::rangeSlider("##clip", &lo, &hi, so)) {
                clipLo_ = lo;
                clipHi_ = hi;
                if (clipChanged) clipChanged(lo, hi);
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
    }

} // namespace sirius::app::gui
