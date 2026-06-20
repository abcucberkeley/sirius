add_library(sirius STATIC)

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(sirius
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_sources(sirius
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ${PROJECT_SOURCE_DIR}/include
        FILES
            ${PROJECT_SOURCE_DIR}/include/sirius/tiff_io.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/sim_parameters.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/legacy_config.hpp
    PRIVATE
        tiff_io.cpp
        fft.cpp
        sim_parameters.cpp
        legacy_config.cpp
)

target_link_libraries(sirius
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        TIFF::TIFF
        OpenMP::OpenMP_CXX
        FFTW3::fftw3
        sirius_simd
        tomlplusplus::tomlplusplus
)

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius)
endif()