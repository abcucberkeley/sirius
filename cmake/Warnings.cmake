# target_set_warnings(<tgt>): the warning set for SIRIUS's own targets (the
# sirius library, the app core, the app, tests, benchmarks; src/, app/, tests/
# and benchmarks/ CMakeLists call it, FetchContent dependencies never do).
# SIRIUS_WARNINGS_AS_ERRORS adds /WX or -Werror to the library and the app
# core only -- that is, to the units they are made of (cmake/Units.cmake:
# sirius_lib_<unit>, sirius_core_<unit>) -- so a new warning in the code CI
# compiles is a build failure while the tests and the Qt layer keep building.
function(target_set_warnings tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
  else()
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
    )
  endif()

  if(SIRIUS_WARNINGS_AS_ERRORS AND tgt MATCHES "^(sirius|sirius_app_core|sirius_(lib|core)_.+)$")
    if(MSVC)
      target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/WX>)
    else()
      target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
    endif()
  endif()
endfunction()
