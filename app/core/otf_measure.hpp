#ifndef SIRIUS_APP_OTF_MEASURE_HPP
#define SIRIUS_APP_OTF_MEASURE_HPP

// Measuring an OTF from a bead stack that is already open as a dataset: the
// one service the dialog, the tool API and the Python bindings all call.
//
// sirius/otf_measure.hpp is the measurement itself and takes a bare
// (sections, ny, nx) tensor with every number stated. This layer is what turns
// "the dataset in front of me" into those numbers: the pixel sizes and the
// phase count come from the acquisition's own metadata, the frames are
// gathered through the dataset's SIM layout rather than assumed to be in the
// library's packing order, the table is written where the user asked with the
// provenance that ties it back to the stack, and the result carries the
// Diagnostics the dialog draws.
//
// THE SHAPE IS app/core/training_export.hpp's, deliberately: an options
// struct, ONE validate that answers before anything is read, and a run
// function returning a result struct. The reason is finding 9k.50 item 5 --
// one condition (an even image size) carried three different error strings and
// three exception types across the GUI, the CLI and Python, and nothing at all
// in Python. So the words for a condition live HERE, once:
// validateOtfMeasureRequest returns every problem it found, each tagged with
// the field it belongs to, and otfMeasureProblems joins them for a caller that
// has room for one line. A front end that invents its own wording for a
// condition in this list is the defect, not a convenience.
//
// WHAT THE DATASET CAN ANSWER AND WHAT IT CANNOT. otfMeasureDefaults fills
// everything the acquisition states -- dxy, dz, the phase count, the section
// order, the saturation level implied by the pixel type -- and leaves the rest
// at the library's defaults. Three numbers the dataset genuinely does not
// know, and which therefore stay the user's to state: the bead diameter, and
// the illumination line spacing and angle that the finite-bead-size division
// applies to the side bands (sirius/otf_measure.hpp, beadDiameterUm /
// patternPeriodUm / patternAngleRad). They are not guessed from the data here,
// because a wrong guess is a wrong table with nothing saying so, which is the
// failure mode of 9k.48 one layer up. The line spacing's own default is 0 --
// NOT STATED -- rather than a plausible number, so a measurement made without
// it says so in its notes, in its provenance
// (options.pattern_period_stated) and in its summary instead of quietly using
// makeotf's 0.2 um on an instrument that runs 0.504 um. A caller holding the
// acquisition's SIM parameters states it by calling
// sirius::OtfMeasureOptions::setIllumination on request.measure.
//
// THE OUTPUT PATH IS NOT DEFAULTED NEXT TO THE STACK. otfMeasureFileName
// proposes a NAME ("OTF_488_sirius.tif", from the channel's own wavelength)
// and nothing proposes a folder, so no caller can write a calibration artefact
// into the acquisition tree it was measured from by accepting a default. On
// this project's clusters those trees are read-only by rule.
//
// THE SECTION ORDER IS READ, NOT ASSUMED. A raw SIM stack packs (angle, phase,
// z) along one file axis and SimLayout says how (core/dataset.hpp). When the
// dataset carries a layout that binds to its dims, the frames of ONE
// illumination direction are gathered through SimFrames::frameOf -- so a
// montage, a phase-slowest file or the angles on the channel axis all arrive
// as the phase-fastest stack the library documents -- and
// OtfMeasureReport::sectionOrder says Layout. Without a layout the z axis is
// handed over as it is stored and the request's packing decides, and
// sectionOrder says Packing. The two cannot disagree silently: a request whose
// packing or nphases contradicts the layout is a validation problem naming
// both numbers, not an override.
//
// ONE DIRECTION. An OTF is measured per illumination direction (makeotf takes
// one stack and one k0), so a multi-direction acquisition measures the
// direction `angle` names and the report says which. Measuring all of them is
// several calls and several tables, which is also what cudasirecon's otfPerAngle
// expects.
//
// THE DIAGNOSTICS ARE THE EXISTING TYPES (core/diagnostics.hpp) and no new
// ones: images, a table, curves, facts, warnings. That is what lets the next
// stage's dialog draw this report with the cells the diagnostics dock already
// has -- app/imgui/panels/diagnostic_cells.hpp states in its own first
// paragraph that none of it knows the workbench, so a dialog or a report can
// reuse every cell by handing it a Diagnostics. Worth knowing when reading the
// report: DiagnosticsBody's Generic body draws images 0 and 1, the facts, the
// curves and the histograms, but it draws `table` only for DiagnosticsKind::Sim,
// so the bead inventory below is for the dialog to draw with cells::table
// directly. Image 0 is therefore the bead field with its candidates marked and
// image 1 the measured order 0, which is the pair worth seeing first.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/otf_measure.hpp>

#include "core/array.hpp"
#include "core/dataset.hpp"
#include "core/diagnostics.hpp"

namespace sirius::app {

    // How the sections handed to the library were put in order.
    enum class OtfSectionOrder {
        Layout,   // gathered through the dataset's SIM layout (one direction, phase fastest)
        Packing   // the z axis as the file stores it, de-interleaved by the request's packing
    };
    const char* otfSectionOrderName(OtfSectionOrder o) noexcept;

    // The user's choices plus the library's options. Named Request rather than
    // Options because it carries sirius::OtfMeasureOptions whole: the library
    // owns the measurement's parameters and this adds only what a dataset and
    // a destination contribute, so there is one copy of each parameter and no
    // mapping layer to fall out of step.
    struct OtfMeasureRequest {
        sirius::OtfMeasureOptions measure;   // the measurement itself

        // --- which part of the dataset
        Index channel = 0;        // real channel (not a SIM axis)
        Index time = 0;           // real time point
        Index angle = 0;          // illumination direction, when the layout has several

        // --- what to write; an empty path measures and writes nothing
        std::string path;         // the .tif ("otf.tif"); ".tif" is appended when missing
        bool sidecar = true;      // <path>.toml, the sampling loadOTF prefers to any derivation
        bool provenanceFile = true;   // <path>.json, the provenance below
        bool overwrite = false;   // refuse an existing file unless this is set
        std::string note;         // free text into the sidecar and the provenance

        Index previewMaxSide = 192;   // the bead-field preview's longer side, px
    };

    // One failed condition. `field` names the control it belongs to, so a
    // dialog can mark it and a tool can say which argument was wrong, without
    // either of them rewording the message.
    struct OtfMeasureProblem {
        std::string field;     // "dxy", "nphases", "path", "angle", "packing", ...
        std::string message;   // the one wording for this condition
    };

    // THE validate. Empty when the request fits the dataset. It answers from
    // the metadata alone -- no voxel is read -- so a dialog can call it on
    // every keystroke and a tool can refuse before materialising an array.
    // The library's own validateOtfMeasure is called from inside it with the
    // shape the gather will produce, so its messages reach a caller here too
    // rather than only as an exception from the measurement.
    std::vector<OtfMeasureProblem> validateOtfMeasureRequest(const OtfMeasureRequest& request, const DatasetMeta& meta,
                                                             const Dims5& dims);
    // "field: message" per problem, "; " between them. Empty for no problems.
    std::string otfMeasureProblems(const std::vector<OtfMeasureProblem>& problems);

    // The request this dataset implies: everything its metadata states, with
    // the library's defaults for everything it does not. `path` is left empty
    // (see the header's note on the output path).
    OtfMeasureRequest otfMeasureDefaults(const DatasetMeta& meta, const Dims5& dims);

    // A file name for the table, from the channel's own wavelength:
    // "OTF_488_sirius.tif", or "OTF_sirius.tif" when the channel says nothing.
    // A name only -- the folder is the caller's.
    std::string otfMeasureFileName(const DatasetMeta& meta, Index channel);

    struct OtfMeasureReport {
        sirius::OtfMeasureResult measurement;   // the library's result, whole

        // --- what the gather did
        OtfSectionOrder sectionOrder = OtfSectionOrder::Packing;
        int sections = 0, nphases = 0, nz = 0;
        Index angles = 1;              // directions the layout holds; the one measured is request.angle
        Index ny = 0, nx = 0;          // the section the measurement saw (a montage tile's, if any)

        // --- what was written, the table first; empty when path was empty
        std::vector<std::string> files;
        std::filesystem::path tablePath;
        std::uint64_t bytes = 0;

        // --- for the user and for the record
        nlohmann::json provenance;     // the stack, the parameters, the beads, the sampling
        Diagnostics diagnostics;
        std::string summary;           // one line; the long form is measurement.summary()
        std::vector<std::string> notes;   // this layer's own, beside measurement.notes
    };

    // Measures, writes what the request asks for, and reports. Throws
    // std::invalid_argument with otfMeasureProblems' text when the request
    // does not fit (so a caller that skipped validate still cannot proceed on
    // a bad one), CancelledError when `cancelled` says so, and whatever the
    // library or the file system throws otherwise.
    OtfMeasureReport measureOtfFromDataset(const Array5& array, const DatasetMeta& meta,
                                           const OtfMeasureRequest& request,
                                           const std::function<void(double, const std::string&)>& progress = {},
                                           const std::function<bool()>& cancelled = {});

    // The diagnostics of a finished measurement, so a dialog can redraw a
    // report it kept without measuring again. `preview` becomes image 0 when
    // it is given (the stack it needs is not in the result).
    Diagnostics otfMeasureDiagnostics(const sirius::OtfMeasureResult& result, const OtfMeasureRequest& request,
                                      const DiagnosticImage* preview = nullptr);

    // The provenance object, likewise rebuildable without measuring again.
    nlohmann::json otfMeasureProvenance(const sirius::OtfMeasureResult& result, const DatasetMeta& meta,
                                        const OtfMeasureRequest& request, const OtfMeasureReport& report);

    // --- the JSON face the tool and the bindings share --------------------------
    //
    // The tool API and a Python binding both receive their arguments as JSON
    // and both have to turn them into an OtfMeasureRequest. Doing that twice
    // is the same defect one layer down from the one validate above: two
    // places deciding what "scale" may be, and two wordings when it is wrong.
    // So the parsing lives here, the tool's handler is a few lines over it,
    // and the only thing a front end states for itself is the schema.
    //
    // Every key is optional and names a field of OtfMeasureRequest in
    // snake_case: channel, time, angle, path, note, overwrite, sidecar,
    // provenance, preview_max_side, nphases, norders, packing, phases, dxy,
    // dz, background, background_estimate, background_border,
    // darkest_fraction, apodize, bead_diameter_um, pattern_period_um,
    // pattern_angle_rad, bead_compensation_pixel_um,
    // bead_compensation_axial_um, scale, line_fit_first, line_fit_last,
    // band_ratio_min_order0, repair_kr0_column, combine_reim, field,
    // per_bead_normalise, and a nested `detect` object for
    // BeadDetectionOptions. What is absent keeps otfMeasureDefaults' answer,
    // so a caller states only what the acquisition cannot. The three enum
    // keys take "phase_fastest" / "phase_slowest", "order0_dc" /
    // "makeotf_fixorigin" / "as_measured" and "border_mean" /
    // "darkest_fraction"; anything else throws std::invalid_argument naming
    // the key and what it accepts.
    //
    // ANY OTHER KEY IS REFUSED, top level or inside `detect`, and so is a
    // `detect` that is not an object. This face used to ignore what it did not
    // recognise, which turns a caller's typo into a measurement on the
    // defaults reported as a success -- the same silence as an unstated
    // constant. A front end that wraps these arguments in an envelope of its
    // own (a dataset id, a request id) therefore passes the measurement's own
    // object, not the envelope.
    OtfMeasureRequest otfMeasureRequestFromJson(const nlohmann::json& args, const DatasetMeta& meta, const Dims5& dims);

    // The report as a tool reply: what was written, the table's shape and
    // sampling, the bead counts, the scale with its softness, the warnings a
    // user has to read before quoting a depth, and the provenance whole (so a
    // caller that wrote no file still has it). The images and curves are not
    // in it -- a reply is text, and a front end that wants them has the
    // Diagnostics.
    nlohmann::json otfMeasureReportJson(const OtfMeasureReport& report);

} // namespace sirius::app

#endif // SIRIUS_APP_OTF_MEASURE_HPP
