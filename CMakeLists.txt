cmake_minimum_required(VERSION 3.15)
project(pylibjxl LANGUAGES C CXX)

# Optimization: Enable IPO and Hidden Visibility as per GEMINI.md
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Configure libjpeg-turbo options
set(ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(ENABLE_STATIC ON CACHE BOOL "" FORCE)
set(WITH_TURBOJPEG ON CACHE BOOL "" FORCE)
set(WITH_TOOLS OFF CACHE BOOL "" FORCE)
set(WITH_TESTS OFF CACHE BOOL "" FORCE)

# Fix for Linux LTO: Disable TLS to avoid relocation R_X86_64_TPOFF32 error
if(UNIX AND NOT APPLE)
    set(WITH_TLS OFF CACHE BOOL "" FORCE)
endif()

# Prevent third-party libraries from installing their files into our wheel
set(SKIP_INSTALL_ALL TRUE)
set(SKIP_INSTALL_LIBRARIES TRUE)
set(SKIP_INSTALL_HEADERS TRUE)
set(SKIP_INSTALL_EXECUTABLES TRUE)
set(SKIP_INSTALL_FILES TRUE)

# Apply patch to libjpeg-turbo
find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --check ${CMAKE_CURRENT_SOURCE_DIR}/patches/libjpeg-turbo.patch
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo
    RESULT_VARIABLE patch_check_result
    OUTPUT_QUIET ERROR_QUIET
)
if(patch_check_result EQUAL 0)
    message(STATUS "Applying libjpeg-turbo patch...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/libjpeg-turbo.patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo
        RESULT_VARIABLE patch_apply_result
    )
    if(NOT patch_apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply libjpeg-turbo patch")
    endif()
else()
    message(STATUS "libjpeg-turbo patch seems to be already applied or inapplicable.")
endif()

add_subdirectory(third_party/libjpeg-turbo)

# Fix for Linux LTO: Disable IPO for third-party targets to avoid linkage issues
if(ipo_supported AND UNIX AND NOT APPLE)
    if(TARGET turbojpeg-static)
        set_target_properties(turbojpeg-static PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    if(TARGET jpeg-static)
        set_target_properties(jpeg-static PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    if(TARGET simd)
        set_target_properties(simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
endif()

# Expose libjpeg-turbo to libjxl (mock FindJPEG)
set(JPEG_FOUND TRUE CACHE BOOL "" FORCE)
set(JPEG_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/src"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/libjpeg-turbo") # for jconfig.h
set(JPEG_LIBRARIES turbojpeg-static)
set(JPEG_LIBRARY ${JPEG_LIBRARIES} CACHE STRING "" FORCE)
set(JPEG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/src" CACHE PATH "" FORCE)

# Configure libjxl options
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_MANPAGES OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_BENCHMARK OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_FUZZERS OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_SKCMS ON CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_VIEWERS OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(JPEGXL_FORCE_SYSTEM_BROTLI OFF CACHE BOOL "" FORCE)
set(JPEGXL_FORCE_SYSTEM_HWY OFF CACHE BOOL "" FORCE)
set(JPEGXL_FORCE_SYSTEM_LCMS2 OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TRANSCODE_JPEG ON CACHE BOOL "" FORCE)

# Add libjxl
add_subdirectory(third_party/libjxl)

# Find pybind11
find_package(pybind11 REQUIRED)

# Define the extension module
pybind11_add_module(_pylibjxl src/main.cpp)

# Link with libjxl and turbojpeg
# Ensure headers are found globally
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/src"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/libjpeg-turbo"
)

target_link_libraries(_pylibjxl PRIVATE jxl jxl_threads turbojpeg-static)

# Set target properties
target_compile_features(_pylibjxl PRIVATE cxx_std_17)

# Assign to runtime component so scikit-build-core only installs this
install(TARGETS _pylibjxl
    DESTINATION pylibjxl
    COMPONENT runtime
)

# Strip symbols in Release mode to further reduce size
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
        add_custom_command(TARGET _pylibjxl POST_BUILD
            COMMAND strip -x $<TARGET_FILE:_pylibjxl>
            COMMENT "Stripping symbols from _pylibjxl"
        )
    elseif(UNIX)
        add_custom_command(TARGET _pylibjxl POST_BUILD
            COMMAND strip --strip-unneeded $<TARGET_FILE:_pylibjxl>
            COMMENT "Stripping symbols from _pylibjxl"
        )
    endif()
endif()
