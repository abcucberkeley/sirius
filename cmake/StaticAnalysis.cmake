if(SIRIUS_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "SIRIUS_ENABLE_CLANG_TIDY is ON but clang-tidy was not found — static analysis will not run")
  endif()
endif()

if(SIRIUS_ENABLE_CPPCHECK)
  find_program(CPPCHECK_EXE NAMES cppcheck)
  if(CPPCHECK_EXE)
    set(CMAKE_CXX_CPPCHECK
      "${CPPCHECK_EXE};--enable=warning,performance,portability;--inline-suppr")
  else()
    message(WARNING "SIRIUS_ENABLE_CPPCHECK is ON but cppcheck was not found — static analysis will not run")
  endif()
endif()

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/analyze>
    # Don't run code analysis on external/system headers (Eigen, toml++,
    # libtiff, ...). Their header-only code produces false positives we can't
    # fix (e.g. toml++ C6011/C28199, Eigen C6326). Dependency include dirs are
    # marked SYSTEM (emitted as /external:I), which this option then skips.
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/analyze:external->
  )
endif()