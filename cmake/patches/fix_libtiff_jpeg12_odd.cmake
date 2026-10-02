# Patch for an upstream libtiff bug in 12-bit JPEG decoding (present in 4.7.0
# and still on master as of this pin).
#
# JPEGDecode() repacks the 16-bit samples libjpeg-turbo returns for a 12-bit
# image into libtiff's packed 12-bit rows two samples (three bytes) at a time,
# `for (iPair = 0; iPair < width * components / 2; ...)`, so a row with an odd
# number of samples -- a 37-pixel-wide greyscale strip, say -- loses its last
# sample: it reads as 0. tifffile (imagecodecs) decodes it; SIRIUS must too.
# The patch packs the remaining sample into the byte and a half that the row
# reserves for it.
#
# Run via FetchContent's PATCH_COMMAND, whose working directory is the
# populated libtiff source tree. Applying it twice is a no-op.

set(_file "libtiff/tif_jpeg.c")
if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "fix_libtiff_jpeg12_odd: ${_file} not found. Has the libtiff layout changed?")
endif()
file(READ "${_file}" _contents)
string(REPLACE "\r\n" "\n" _contents "${_contents}")   # a checkout with CRLF line ends
if(_contents MATCHES "SIRIUS patch: odd 12-bit")
    return()
endif()

set(_anchor [=[
                        out_ptr[2] = (unsigned char)(((in_ptr[1] & 0xff) >> 0));
                    }
                }
                else if (sp->cinfo.d.data_precision == 8)]=])
set(_patched [=[
                        out_ptr[2] = (unsigned char)(((in_ptr[1] & 0xff) >> 0));
                    }
                    /* SIRIUS patch: odd 12-bit sample count -- pack the last
                       sample into the byte and a half left for it */
                    if ((sp->cinfo.d.output_width *
                         sp->cinfo.d.num_components) & 1)
                    {
                        unsigned char *out_ptr =
                            ((unsigned char *)buf) + value_pairs * 3;
                        TIFF_JSAMPLE *in_ptr = line_work_buf + value_pairs * 2;
                        out_ptr[0] = (unsigned char)((in_ptr[0] & 0xff0) >> 4);
                        out_ptr[1] = (unsigned char)((in_ptr[0] & 0xf) << 4);
                    }
                }
                else if (sp->cinfo.d.data_precision == 8)]=])
string(FIND "${_contents}" "${_anchor}" _at)
if(_at EQUAL -1)
    message(FATAL_ERROR "fix_libtiff_jpeg12_odd: the 12-bit repacking loop in ${_file} has changed; "
                        "check whether the bug is fixed upstream and drop or update this patch.")
endif()
string(REPLACE "${_anchor}" "${_patched}" _contents "${_contents}")
file(WRITE "${_file}" "${_contents}")
# The 12-bit codec is tif_jpeg_12.c, which #includes tif_jpeg.c; not every
# generator tracks that include, so mark it changed too.
if(EXISTS "libtiff/tif_jpeg_12.c")
    file(TOUCH "libtiff/tif_jpeg_12.c")
endif()
message(STATUS "fix_libtiff_jpeg12_odd: patched ${_file} (last sample of odd 12-bit JPEG rows)")
