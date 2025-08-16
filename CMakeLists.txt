cmake_minimum_required(VERSION 3.16)
project(VisionForge LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Default to Release if not provided ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Warnings ----
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Header-only interface (include paths) ----
if(NOT TARGET visionforge)
  add_library(visionforge INTERFACE)
  target_include_directories(visionforge INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
endif()

# ---- TinyEXR I/O ----
if(NOT TARGET vf_io)
  add_library(vf_io STATIC
    src/io/exr_writer.cpp
  )
  target_include_directories(vf_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
  find_package(ZLIB REQUIRED)
  target_link_libraries(vf_io PUBLIC ZLIB::ZLIB)
endif()

# ---- Main executable ----
if(NOT TARGET visionforge_app)
  add_executable(visionforge_app apps/visionforge/main.cpp)
  target_link_libraries(visionforge_app PRIVATE visionforge vf_io)
  set_target_properties(visionforge_app PROPERTIES OUTPUT_NAME "visionforge")
endif()

# ---- Optimizations (non‑MSVC) ----
if (NOT MSVC)
  target_compile_options(vf_io           PRIVATE -O3 -march=native -DNDEBUG)
  target_compile_options(visionforge_app PRIVATE -O3 -march=native -DNDEBUG)
endif()

# ---- OpenMP (parallel render) ----
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(visionforge_app PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(visionforge_app PUBLIC VF_USE_OMP=1)
endif()

# ---- (Optional) IPO/LTO ----
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if (ipo_ok)
  set_property(TARGET visionforge_app PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_property(TARGET vf_io            PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
