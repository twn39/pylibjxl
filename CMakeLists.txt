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

# Configure libjxl options before adding it
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

# Add libjxl
add_subdirectory(third_party/libjxl)

# Find pybind11
find_package(pybind11 REQUIRED)

# Define the extension module
pybind11_add_module(_pylibjxl src/main.cpp)

# Link with libjxl
target_link_libraries(_pylibjxl PRIVATE jxl jxl_threads)

# Set target properties
target_compile_features(_pylibjxl PRIVATE cxx_std_17)

install(TARGETS _pylibjxl DESTINATION pylibjxl)
