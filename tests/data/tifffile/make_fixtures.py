"""Write the TIFF fixtures in this folder with tifffile (test reference only).

SIRIUS reads TIFF with its own C++ reader; these small files, written by an
independent writer, let tests/test_tiff_formats.cpp check that reader against
files it did not write. Every array holds the same pattern the C++ test
computes (pattern() below == rawValue / floatValue there), so the test needs
no copy of the arrays. Run from anywhere:

    python tests/data/tifffile/make_fixtures.py

Needs tifffile and imagecodecs (LZW, Deflate predictors, ZSTD, JPEG). The
files are checked in; re-running rewrites them byte for byte only with the
same tifffile version, which does not matter: the content is what is tested.

JPEG is lossy, so no pattern predicts its pixels: each JPEG fixture comes with
<name>.expected.raw, what tifffile + imagecodecs (libjpeg-turbo) decode it
to, as (pages, samples, height, width) little-endian samples -- the C++ test
compares SIRIUS's decode with those bytes exactly.
"""

from __future__ import annotations

import os

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))


def pattern(pages: int, samples: int, h: int, w: int, bits: int = 16, kind: str = "u") -> np.ndarray:
    """(pages, samples, h, w): p*131 + s*37 + y*7 + x*3 + (x*y) % 11 -- the
    C++ test's rawValue -- reduced to `bits` bits, or (v - 40) / 4 for floats."""
    p, s, y, x = np.meshgrid(np.arange(pages), np.arange(samples), np.arange(h), np.arange(w), indexing="ij")
    v = (p * 131 + s * 37 + y * 7 + x * 3 + (x * y) % 11).astype(np.int64)
    if kind == "f":
        return (v.astype(np.float64) - 40.0) * 0.25
    return v & ((1 << bits) - 1)


def path(name: str) -> str:
    return os.path.join(HERE, name)


def write_expected(name: str) -> None:
    """tifffile's decode of fixture `name`, as SIRIUS lays it out, to <name>.expected.raw."""
    with tifffile.TiffFile(path(name)) as t:
        a = np.stack([p.asarray() for p in t.pages])
        if a.ndim == 4:   # contiguous samples: (pages, h, w, s) -> (pages, s, h, w)
            a = np.moveaxis(a, -1, 1)
    a.astype(a.dtype.newbyteorder("<")).tofile(path(name.replace(".tif", ".expected.raw")))


def main() -> None:
    h, w = 29, 37
    # RGB, contiguous samples, LZW + horizontal predictor, strips
    rgb = pattern(2, 3, h, w, 8).astype(np.uint8)
    tifffile.imwrite(path("rgb_contig_lzw.tif"), np.moveaxis(rgb, 1, -1), photometric="rgb",
                     compression="lzw", predictor=True, rowsperstrip=5)
    # RGB, separate planes, Deflate, 16x16 tiles, uint16
    rgb16 = pattern(2, 3, h, w, 16).astype(np.uint16)
    tifffile.imwrite(path("rgb_planar_tiled_deflate.tif"), rgb16, photometric="rgb", planarconfig="separate",
                     compression="zlib", tile=(16, 16))
    # big-endian float32 with the floating-point predictor
    f32 = pattern(2, 1, h, w, kind="f").astype(np.float32)[:, 0]
    tifffile.imwrite(path("be_float32_fppred.tif"), f32, byteorder=">", compression="zlib", predictor=3,
                     photometric="minisblack")
    # float16, tiled
    f16 = pattern(2, 1, h, w, kind="f").astype(np.float16)[:, 0]
    tifffile.imwrite(path("float16_tiled.tif"), f16, tile=(16, 16), photometric="minisblack")
    # PackBits, int16 (negative values after the shift)
    i16 = (pattern(2, 1, h, w, 16)[:, 0] - 300).astype(np.int16)
    tifffile.imwrite(path("int16_packbits.tif"), i16, compression="packbits", photometric="minisblack")
    # bilevel (1-bit) image
    b1 = (pattern(1, 1, h, w, 1)[0, 0]).astype(bool)
    tifffile.imwrite(path("bilevel.tif"), b1)
    # ZSTD: uint16 tiles with the horizontal predictor, float32 strips with the floating-point one
    tifffile.imwrite(path("zstd_uint16_tiled_pred.tif"), pattern(2, 1, h, w, 16).astype(np.uint16)[:, 0],
                     compression="zstd", predictor=True, tile=(16, 16), photometric="minisblack")
    tifffile.imwrite(path("zstd_float32_fppred.tif"), f32, compression="zstd", predictor=3, rowsperstrip=8,
                     photometric="minisblack")
    # JPEG: RGB stored as 2x2-subsampled YCbCr in tiles (read back as RGB), greyscale strips, 12-bit
    tifffile.imwrite(path("jpeg_ycbcr_tiled.tif"), np.moveaxis(rgb, 1, -1), photometric="rgb", compression="jpeg",
                     subsampling=(2, 2), tile=(16, 16))
    write_expected("jpeg_ycbcr_tiled.tif")
    tifffile.imwrite(path("jpeg_grey_strips.tif"), pattern(2, 1, h, w, 8).astype(np.uint8)[:, 0],
                     photometric="minisblack", compression="jpeg", rowsperstrip=16)
    write_expected("jpeg_grey_strips.tif")
    tifffile.imwrite(path("jpeg12_grey.tif"), pattern(2, 1, h, w, 12).astype(np.uint16)[:, 0],
                     photometric="minisblack", compression="jpeg", bitspersample=12)
    write_expected("jpeg12_grey.tif")
    # ImageJ hyperstack: 3 t, 2 z, 2 c, with spacing, unit and frame interval
    ij = pattern(12, 1, h, w, 16)[:, 0].astype(np.uint16).reshape(3, 2, 2, h, w)
    tifffile.imwrite(path("imagej_hyperstack.tif"), ij, imagej=True, resolution=(1 / 0.25, 1 / 0.25),
                     metadata={"axes": "TZCYX", "spacing": 0.5, "unit": "um", "finterval": 2.0})
    # OME-TIFF: 2 z of an RGB image, with physical sizes and a channel name
    ome_rgb = np.moveaxis(pattern(2, 3, h, w, 8).astype(np.uint8), 1, -1)
    tifffile.imwrite(path("ome_rgb.ome.tif"), ome_rgb, photometric="rgb",
                     metadata={"axes": "ZYXS", "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.3,
                               "Channel": {"Name": ["RGB"]}})
    # OME-TIFF with two series of different shapes, the first with a 2-level SubIFD pyramid
    s0 = pattern(3, 1, 64, 48, 16)[:, 0].astype(np.uint16)
    s1 = pattern(2, 1, h, w, 16)[:, 0].astype(np.uint16) + 7
    with tifffile.TiffWriter(path("ome_series_pyramid.ome.tif"), ome=True) as tw:
        tw.write(s0, subifds=1, tile=(16, 16), photometric="minisblack",
                 metadata={"axes": "ZYX", "Name": "big", "PhysicalSizeX": 0.2, "PhysicalSizeY": 0.2})
        tw.write(s0[:, ::2, ::2], subfiletype=1, tile=(16, 16), photometric="minisblack")
        tw.write(s1, photometric="minisblack", metadata={"axes": "ZYX", "Name": "small"})


if __name__ == "__main__":
    main()
