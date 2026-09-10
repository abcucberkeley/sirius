#include "qt/viewer/volume_view.hpp"

#include "core/labels.hpp"

#include <sirius/constants.hpp>

#include <algorithm>
#include <cmath>

#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QPainter>
#include <QSlider>
#include <QVector3D>
#include <QWheelEvent>

#include "qt/theme.hpp"
#include "qt/viewer/viewer_constants.hpp"
#include "qt/viewer/viewer_widgets.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {
        using viewer::kVolumeMaxChannels;
        constexpr int kMaxChannels = kVolumeMaxChannels;

        const char* kVertex = R"(
            out vec2 vNdc;
            void main() {
                // full-screen triangle from gl_VertexID, no buffers needed
                vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
                vNdc = p;
                gl_Position = vec4(p, 0.0, 1.0);
            })";

        const char* kFragment = R"(
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

        const char* kLineVertex = R"(
            in vec3 aPos;
            uniform mat4 uViewProj;
            void main() { gl_Position = uViewProj * vec4(aPos, 1.0); })";

        const char* kLineFragment = R"(
            out vec4 fragColor;
            uniform vec4 uColor;
            void main() { fragColor = uColor; })";

        QString versionPrefix(QOpenGLContext* ctx) {
            if (ctx->isOpenGLES()) return QStringLiteral("#version 300 es\nprecision highp float;\nprecision highp sampler3D;\n");
            return QStringLiteral("#version 330 core\n");
        }
    } // namespace

    struct VolumeView::Gl {
        std::unique_ptr<QOpenGLShaderProgram> ray, line;
        QOpenGLVertexArrayObject vao;
        GLuint textures[kMaxChannels] = {0, 0, 0, 0};
        int textureCount = 0;
        GLuint labelTexture = 0;
        bool labelsUploaded = false;
        GLuint lineVbo = 0;
    };

    // Child widgets laid over the GL surface: presets, yaw / pitch sliders, clip.
    class VolumeView::Overlays {
    public:
        explicit Overlays(VolumeView* view) : view(view) {
            static const struct {
                const char* name;
                double yaw, pitch;
            } presets[] = {
                {"Front", 0, 0}, {"Iso", 35, 22}, {"Top", 0, 60}, {"Side", 90, 0}};
            presetHost = new QWidget(view);
            auto* ph = new QHBoxLayout(presetHost);
            ph->setContentsMargins(0, 0, 0, 0);
            ph->setSpacing(8);
            for (const auto& p : presets) {
                auto* b = new widgets::GlyphButton(QString::fromLatin1(p.name), QSize(52, 22), presetHost);
                b->setGlyphPx(11);
                b->setOnDark(true);
                b->setCheckable(true);
                b->setAutoExclusive(false);
                const double yaw = p.yaw, pitch = p.pitch;
                QObject::connect(b, &QAbstractButton::clicked, view, [this, yaw, pitch] { this->view->applyOrientation(yaw, pitch, true); });
                presetButtons.push_back({b, yaw, pitch});
                ph->addWidget(b);
            }
            presetHost->adjustSize();

            sliderHost = new QWidget(view);
            auto* grid = new QGridLayout(sliderHost);
            grid->setContentsMargins(0, 0, 0, 0);
            grid->setHorizontalSpacing(10);
            grid->setVerticalSpacing(6);
            auto label = [&](const QString& t) {
                return widgets::label(t, 11, theme::kViewerText, -1, sliderHost);
            };
            yaw = new QSlider(Qt::Horizontal, sliderHost);
            yaw->setRange(0, 359);
            yaw->setFixedWidth(140);
            pitch = new QSlider(Qt::Horizontal, sliderHost);
            pitch->setRange(-60, 60);
            pitch->setFixedWidth(140);
            for (QSlider* s : {yaw, pitch}) s->setFocusPolicy(Qt::NoFocus);
            grid->addWidget(label(QStringLiteral("Yaw")), 0, 0);
            grid->addWidget(yaw, 0, 1);
            grid->addWidget(label(QStringLiteral("Pitch")), 1, 0);
            grid->addWidget(pitch, 1, 1);
            sliderHost->adjustSize();
            QObject::connect(yaw, &QSlider::valueChanged, view, [this](int v) {
                if (!syncing) this->view->applyOrientation(v, this->view->pitch(), true);
            });
            QObject::connect(pitch, &QSlider::valueChanged, view, [this](int v) {
                if (!syncing) this->view->applyOrientation(this->view->yaw(), v, true);
            });

            clipHost = new QWidget(view);
            auto* ch = new QHBoxLayout(clipHost);
            ch->setContentsMargins(0, 0, 0, 0);
            ch->setSpacing(8);
            auto* cl = label(QStringLiteral("Clip Z"));
            clip = new RangeSlider(clipHost);
            ch->addWidget(cl);
            ch->addWidget(clip);
            clipHost->adjustSize();
            QObject::connect(clip, &RangeSlider::rangeChanged, view, [this](double lo, double hi) {
                this->view->clipLo_ = lo;
                this->view->clipHi_ = hi;
                this->view->update();
                emit this->view->clipChanged(lo, hi);
            });
            for (QWidget* w : {presetHost, sliderHost, clipHost}) w->setAttribute(Qt::WA_TranslucentBackground);
        }

        void sync() {
            syncing = true;
            yaw->setValue(static_cast<int>(std::lround(view->yaw_)));
            pitch->setValue(static_cast<int>(std::lround(view->pitch_)));
            clip->setRange(view->clipLo_, view->clipHi_);
            for (auto& p : presetButtons)
                p.button->setChecked(std::abs(p.yaw - view->yaw_) < 0.5 && std::abs(p.pitch - view->pitch_) < 0.5);
            syncing = false;
        }

        void layout(int w, int h) {
            presetHost->move(10, h - 10 - presetHost->height());
            sliderHost->move(w - 10 - sliderHost->width(), h - 10 - sliderHost->height());
            clipHost->move(w - 10 - clipHost->width(), 8);
        }

        VolumeView* view;
        QWidget* presetHost = nullptr;
        QWidget* sliderHost = nullptr;
        QWidget* clipHost = nullptr;
        QSlider* yaw = nullptr;
        QSlider* pitch = nullptr;
        RangeSlider* clip = nullptr;
        struct Preset {
            widgets::GlyphButton* button;
            double yaw, pitch;
        };
        std::vector<Preset> presetButtons;
        bool syncing = false;
    };

    VolumeView::VolumeView(QWidget* parent) : QOpenGLWidget(parent), gl_(std::make_unique<Gl>()) {
        QSurfaceFormat fmt = QSurfaceFormat::defaultFormat();
        if (fmt.renderableType() != QSurfaceFormat::OpenGLES) {
            fmt.setVersion(3, 3);
            fmt.setProfile(QSurfaceFormat::CoreProfile);
        }
        setFormat(fmt);
        setMouseTracking(true);
        setMinimumSize(80, 80);
        overlays_ = new Overlays(this);
        overlays_->sync();
    }

    VolumeView::~VolumeView() {
        makeCurrent();
        if (glOk_) {
            glDeleteTextures(kMaxChannels, gl_->textures);
            if (gl_->labelTexture) glDeleteTextures(1, &gl_->labelTexture);
            if (gl_->lineVbo) glDeleteBuffers(1, &gl_->lineVbo);
        }
        gl_->ray.reset();
        gl_->line.reset();
        gl_->vao.destroy();
        doneCurrent();
        delete overlays_;
    }

    // --- state -----------------------------------------------------------------------

    void VolumeView::setTextures(quint64 key, std::vector<ReducedVolume> channels, const std::array<double, 3>& voxelUm,
                                 Index nz, Index ny, Index nx) {
        textures_ = std::move(channels);
        if (textures_.size() > static_cast<std::size_t>(kMaxChannels)) textures_.resize(kMaxChannels);
        key_ = key;
        voxelUm_ = voxelUm;
        vz_ = nz;
        vy_ = ny;
        vx_ = nx;
        preparing_.clear();
        update();
    }

    void VolumeView::clearVolumes() {
        textures_.clear();
        vz_ = vy_ = vx_ = 0;
        key_ = 0;
        update();
    }

    void VolumeView::setPreparing(const QString& text) {
        if (preparing_ == text) return;
        preparing_ = text;
        update();
    }

    void VolumeView::setLabels(quint64 key, const std::uint32_t* labels, Index z, Index y, Index x, float opacity, std::uint32_t only) {
        labels_ = labels;
        labelOnly_ = only;
        lz_ = z;
        ly_ = y;
        lx_ = x;
        labelsKey_ = key;
        labelOpacity_ = opacity;
        update();
    }

    void VolumeView::clearLabels() {
        if (!labels_ && labelsKey_ == 0) return;
        labels_ = nullptr;
        labelsKey_ = 0;
        update();
    }

    void VolumeView::applyOrientation(double yaw, double pitch, bool emitSignal) {
        yaw = std::fmod(yaw, 360.0);
        if (yaw < 0) yaw += 360.0;
        pitch = std::clamp(pitch, -60.0, 60.0);
        if (yaw == yaw_ && pitch == pitch_) return;
        yaw_ = yaw;
        pitch_ = pitch;
        overlays_->sync();
        update();
        if (emitSignal) emit orientationChanged(yaw_, pitch_);
    }

    void VolumeView::setOrientation(double yawDeg, double pitchDeg) { applyOrientation(yawDeg, pitchDeg, false); }

    void VolumeView::setClip(double lo, double hi) {
        clipLo_ = std::clamp(lo, 0.0, 1.0);
        clipHi_ = std::clamp(hi, clipLo_, 1.0);
        overlays_->sync();
        update();
    }

    void VolumeView::setBoundingBox(bool on) {
        box_ = on;
        update();
    }

    void VolumeView::setZoom(double zoom) {
        zoom_ = std::clamp(zoom, viewer::kMinVolumeZoom, viewer::kMaxVolumeZoom);
        update();
    }

    void VolumeView::setTransfer(float lo, float hi, float alpha, float stepVoxels, bool mip) {
        tfLo_ = lo;
        tfHi_ = std::max(hi, lo + 1e-3f);
        tfAlpha_ = std::clamp(alpha, 0.0f, 1.0f);
        stepVoxels_ = std::clamp(stepVoxels, 0.1f, 4.0f);
        mip_ = mip;
        update();
    }

    void VolumeView::setMethodText(const QString& text) {
        method_ = text;
        update();
    }

    QImage VolumeView::grabImage() { return grabFramebuffer(); }

    // --- GL ------------------------------------------------------------------------------

    void VolumeView::initializeGL() {
        initializeOpenGLFunctions();
        QOpenGLContext* ctx = context();
        const QString prefix = versionPrefix(ctx);
        auto build = [&](const char* vs, const char* fs, QString& err) {
            auto prog = std::make_unique<QOpenGLShaderProgram>();
            if (!prog->addShaderFromSourceCode(QOpenGLShader::Vertex, prefix + QString::fromLatin1(vs)) ||
                !prog->addShaderFromSourceCode(QOpenGLShader::Fragment, prefix + QString::fromLatin1(fs)) ||
                !prog->link()) {
                err = prog->log();
                return std::unique_ptr<QOpenGLShaderProgram>();
            }
            return prog;
        };
        const auto fmt = ctx->format();
        if (!ctx->isOpenGLES() && fmt.majorVersion() < 3) {
            glError_ = QStringLiteral("OpenGL 3.0 or newer is required for volume rendering (got %1.%2)")
                           .arg(fmt.majorVersion())
                           .arg(fmt.minorVersion());
            return;
        }
        gl_->ray = build(kVertex, kFragment, glError_);
        if (!gl_->ray) return;
        gl_->line = build(kLineVertex, kLineFragment, glError_);
        if (!gl_->line) return;
        gl_->vao.create();
        glGenTextures(kMaxChannels, gl_->textures);
        glGenTextures(1, &gl_->labelTexture);
        glGenBuffers(1, &gl_->lineVbo);
        glOk_ = true;
        uploadedKey_ = 0;
        uploadedLabelsKey_ = 0;
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
        std::vector<unsigned char> texels(static_cast<std::size_t>(tx) * ty * tz * 4, 0);
        bool any = false;
        for (int z = 0; z < tz; ++z) {
            const Index zz = std::min<Index>(static_cast<Index>(z) * fz + fz / 2, lz_ - 1);
            for (int y = 0; y < ty; ++y) {
                const Index yy = std::min<Index>(static_cast<Index>(y) * fy + fy / 2, ly_ - 1);
                const std::uint32_t* row = labels_ + (zz * ly_ + yy) * lx_;
                unsigned char* out = texels.data() + (static_cast<std::size_t>(z) * ty + y) * tx * 4;
                for (int x = 0; x < tx; ++x) {
                    const Index xx = std::min<Index>(static_cast<Index>(x) * fx + fx / 2, lx_ - 1);
                    const std::uint32_t id = row[xx];
                    if (!id || (labelOnly_ != 0 && id != labelOnly_)) continue;
                    const std::array<float, 3> col = labelColor(id);
                    out[x * 4 + 0] = static_cast<unsigned char>(col[0] * 255.0f);
                    out[x * 4 + 1] = static_cast<unsigned char>(col[1] * 255.0f);
                    out[x * 4 + 2] = static_cast<unsigned char>(col[2] * 255.0f);
                    out[x * 4 + 3] = 255;
                    any = true;
                }
            }
        }
        glBindTexture(GL_TEXTURE_3D, gl_->labelTexture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, tx, ty, tz, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels.data());
        glBindTexture(GL_TEXTURE_3D, 0);
        gl_->labelsUploaded = any;
    }

    void VolumeView::resizeGL(int, int) {}

    void VolumeView::uploadTextures() {
        // The reduction happened on the loader thread: this is the upload.
        gl_->textureCount = 0;
        for (const ReducedVolume& brick : textures_) {
            if (brick.texels.empty() || brick.tx <= 0 || brick.ty <= 0 || brick.tz <= 0) continue;
            glBindTexture(GL_TEXTURE_3D, gl_->textures[gl_->textureCount]);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, brick.tx, brick.ty, brick.tz, 0, GL_RED, GL_UNSIGNED_BYTE,
                         brick.texels.data());
            ++gl_->textureCount;
            if (gl_->textureCount >= kMaxChannels) break;
        }
        glBindTexture(GL_TEXTURE_3D, 0);
        uploadedKey_ = key_;
    }

    void VolumeView::paintGL() {
        QPainter painter(this);
        painter.beginNativePainting();
        glDisable(GL_DEPTH_TEST);
        glClearColor(0x0a / 255.0f, 0x09 / 255.0f, 0x09 / 255.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // physical box: the longest side is 1 world unit
        std::array<double, 3> ext{0.0, 0.0, 0.0};
        int nz = 1;
        if (!textures_.empty() && vx_ > 0 && vy_ > 0 && vz_ > 0) {
            ext = {static_cast<double>(vx_) * voxelUm_[0], static_cast<double>(vy_) * voxelUm_[1],
                   static_cast<double>(vz_) * voxelUm_[2]};
            nz = static_cast<int>(vz_);
        }
        const double longest = std::max({ext[0], ext[1], ext[2], 1e-9});
        const QVector3D half(static_cast<float>(ext[0] / longest / 2), static_cast<float>(ext[1] / longest / 2),
                             static_cast<float>(ext[2] / longest / 2));

        QMatrix4x4 proj, view;
        const float aspect = height() > 0 ? static_cast<float>(width()) / static_cast<float>(height()) : 1.0f;
        proj.perspective(32.0f, aspect, 0.05f, 20.0f);
        const double yaw = yaw_ * sirius::kPi / 180.0, pitch = pitch_ * sirius::kPi / 180.0;
        const float dist = static_cast<float>(2.4 / zoom_);
        const QVector3D cam(static_cast<float>(dist * std::sin(yaw) * std::cos(pitch)),
                            static_cast<float>(dist * std::sin(pitch)),
                            static_cast<float>(dist * std::cos(yaw) * std::cos(pitch)));
        view.lookAt(cam, QVector3D(0, 0, 0), QVector3D(0, 1, 0));
        const QMatrix4x4 viewProj = proj * view;

        if (glOk_ && !textures_.empty()) {
            if (uploadedKey_ != key_) uploadTextures();
            if (uploadedLabelsKey_ != labelsKey_) uploadLabels();
            if (gl_->textureCount > 0) {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                gl_->ray->bind();
                gl_->ray->setUniformValue("uInvViewProj", viewProj.inverted());
                gl_->ray->setUniformValue("uHalf", half);
                gl_->ray->setUniformValue("uClipZ", QVector2D(static_cast<float>(clipLo_), static_cast<float>(clipHi_)));
                gl_->ray->setUniformValue("uCount", gl_->textureCount);
                const float stepWorld = static_cast<float>(stepVoxels_ * (2.0 * half.z() / std::max(nz, 1)));
                gl_->ray->setUniformValue("uStep", std::max(stepWorld, 0.002f));
                gl_->ray->setUniformValue("uRamp", QVector3D(tfLo_, tfHi_, tfAlpha_));
                gl_->ray->setUniformValue("uMip", mip_ ? 1 : 0);
                const char* texNames[kMaxChannels] = {"uTex0", "uTex1", "uTex2", "uTex3"};
                int k = 0;
                for (std::size_t i = 0; i < textures_.size() && k < gl_->textureCount; ++i) {
                    const ReducedVolume& brick = textures_[i];
                    if (brick.texels.empty()) continue;
                    glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(k));
                    glBindTexture(GL_TEXTURE_3D, gl_->textures[k]);
                    gl_->ray->setUniformValue(texNames[k], k);
                    gl_->ray->setUniformValue(QByteArrayLiteral("uColor[") + QByteArray::number(k) + "]",
                                              QVector3D(brick.color[0], brick.color[1], brick.color[2]));
                    ++k;
                }
                const bool labelsOn = gl_->labelsUploaded && labels_ != nullptr;
                gl_->ray->setUniformValue("uLabelsOn", labelsOn ? 1 : 0);
                gl_->ray->setUniformValue("uLabelAlpha", labelOpacity_);
                glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(kMaxChannels));
                glBindTexture(GL_TEXTURE_3D, labelsOn ? gl_->labelTexture : 0);
                gl_->ray->setUniformValue("uLabels", kMaxChannels);
                gl_->vao.bind();
                glDrawArrays(GL_TRIANGLES, 0, 3);
                gl_->vao.release();
                gl_->ray->release();
                glActiveTexture(GL_TEXTURE0);
            }
        }

        if (glOk_ && box_ && !textures_.empty()) {
            // 12 edges; the three from the voxel origin (-x, +y, +z corner) in accent
            const float hx = half.x(), hy = half.y(), hz = half.z();
            const QVector3D o(-hx, hy, hz);
            struct Edge {
                QVector3D a, b;
                bool accent;
            };
            const Edge edges[] = {
                {o, {hx, hy, hz}, true}, {o, {-hx, -hy, hz}, true}, {o, {-hx, hy, -hz}, true}, {{hx, hy, hz}, {hx, -hy, hz}, false}, {{hx, hy, hz}, {hx, hy, -hz}, false}, {{-hx, -hy, hz}, {hx, -hy, hz}, false}, {{-hx, -hy, hz}, {-hx, -hy, -hz}, false}, {{-hx, hy, -hz}, {hx, hy, -hz}, false}, {{-hx, hy, -hz}, {-hx, -hy, -hz}, false}, {{hx, -hy, -hz}, {hx, hy, -hz}, false}, {{hx, -hy, -hz}, {-hx, -hy, -hz}, false}, {{hx, -hy, -hz}, {hx, -hy, hz}, false}};
            gl_->line->bind();
            gl_->line->setUniformValue("uViewProj", viewProj);
            gl_->vao.bind();
            glBindBuffer(GL_ARRAY_BUFFER, gl_->lineVbo);
            const int loc = gl_->line->attributeLocation("aPos");
            glEnableVertexAttribArray(static_cast<GLuint>(loc));
            glVertexAttribPointer(static_cast<GLuint>(loc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<float> verts;
                for (const Edge& e : edges) {
                    if (e.accent != (pass == 1)) continue;
                    verts.insert(verts.end(), {e.a.x(), e.a.y(), e.a.z(), e.b.x(), e.b.y(), e.b.z()});
                }
                if (verts.empty()) continue;
                glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(float)), verts.data(), GL_DYNAMIC_DRAW);
                if (pass == 1) gl_->line->setUniformValue("uColor", QVector4D(0xec / 255.f, 0x30 / 255.f, 0x13 / 255.f, 1.0f));
                else gl_->line->setUniformValue("uColor", QVector4D(0.95f, 0.95f, 0.95f, 0.35f));
                glLineWidth(pass == 1 ? 1.5f : 1.0f);
                glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(verts.size() / 3));
            }
            glDisableVertexAttribArray(static_cast<GLuint>(loc));
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            gl_->vao.release();
            gl_->line->release();
        }
        painter.endNativePainting();

        // overlays on top of the GL surface
        painter.setRenderHint(QPainter::Antialiasing, true);
        const QFontMetrics fm(theme::font(11, QFont::ExtraBold));
        drawOverlayText(painter, QPointF(10, 8), QStringLiteral("VOLUME"), true);
        double x = 10 + fm.horizontalAdvance(QStringLiteral("VOLUME")) + 12;
        drawOverlayText(painter, QPointF(x, 8), method_, false, 0.7);
        x += QFontMetrics(theme::font(11)).horizontalAdvance(method_) + 12;
        drawOverlayText(painter, QPointF(x, 8),
                        QStringLiteral("yaw %1° · pitch %2°").arg(std::lround(yaw_)).arg(std::lround(pitch_)), false, 0.7);
        if (!glOk_) {
            painter.setFont(theme::font(12));
            painter.setPen(QColor(243, 242, 242, 180));
            painter.drawText(rect().adjusted(12, 12, -12, -12), Qt::AlignCenter | Qt::TextWordWrap,
                             QStringLiteral("Volume rendering unavailable: ") + glError_);
        } else if (textures_.empty()) {
            painter.setFont(theme::font(12));
            painter.setPen(QColor(243, 242, 242, preparing_.isEmpty() ? 140 : 200));
            painter.drawText(rect(), Qt::AlignCenter,
                             preparing_.isEmpty() ? QStringLiteral("No volume to render") : preparing_);
        } else if (!preparing_.isEmpty()) {
            drawOverlayText(painter, QPointF(viewer::kOverlayInset, viewer::kOverlayTop + 18), preparing_, false, 0.75);
        }
    }

    // --- interaction --------------------------------------------------------------------

    void VolumeView::mousePressEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton) {
            dragging_ = true;
            dragLast_ = e->position();
        }
    }

    void VolumeView::mouseMoveEvent(QMouseEvent* e) {
        if (!dragging_) return;
        const QPointF d = e->position() - dragLast_;
        dragLast_ = e->position();
        // Drag the volume, not the camera: dragging right turns the volume to
        // the right and dragging down tips its top towards the viewer. The
        // camera orbits at (sin yaw, sin pitch, cos yaw), so both angles move
        // against the drag to make the volume follow the pointer.
        applyOrientation(yaw_ - d.x() * 0.5, pitch_ + d.y() * 0.5, true);
    }

    void VolumeView::mouseReleaseEvent(QMouseEvent*) { dragging_ = false; }

    void VolumeView::wheelEvent(QWheelEvent* e) {
        const double steps = e->angleDelta().y() / 120.0;
        if (steps == 0.0) return;
        setZoom(zoom_ * std::pow(viewer::kWheelZoomBase, steps));
        emit zoomChanged(zoom_);
        e->accept();
    }

    void VolumeView::resizeEvent(QResizeEvent* e) {
        QOpenGLWidget::resizeEvent(e);
        overlays_->layout(width(), height());
    }

} // namespace sirius::app
