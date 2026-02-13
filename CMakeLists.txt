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
add_subdirectory(third_party/libjpeg-turbo)

# Expose libjpeg-turbo to libjxl (mock FindJPEG)
set(JPEG_FOUND TRUE CACHE BOOL "" FORCE)
set(JPEG_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/src"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/libjpeg-turbo") # for jconfig.h
set(JPEG_LIBRARIES turbojpeg-static)
set(JPEG_LIBRARY ${JPEG_LIBRARIES} CACHE STRING "" FORCE)
set(JPEG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo/src" CACHE PATH "" FORCE)

# Configure libjxl options
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
target_link_libraries(_pylibjxl PRIVATE jxl jxl_threads turbojpeg-static)

# Add includes for turbojpeg.h
target_include_directories(_pylibjxl PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/libjpeg-turbo")

# Set target properties
target_compile_features(_pylibjxl PRIVATE cxx_std_17)

install(TARGETS _pylibjxl DESTINATION pylibjxl)
