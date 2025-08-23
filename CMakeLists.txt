cmake_minimum_required(VERSION 3.16)
project(VisionForge VERSION 0.1.0 LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Default to Release if not provided ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options ----
option(VISIONFORGE_OMP "Enable OpenMP parallel rendering if available" ON)
option(VISIONFORGE_LTO "Enable interprocedural optimization (LTO) if supported" ON)

# ---- Warnings ----
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Header-only "engine" include paths ----
add_library(visionforge INTERFACE)
target_include_directories(visionforge INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ---- IO (TinyEXR or your wrapper) ----
add_library(vf_io STATIC
  src/io/exr_writer.cpp
)
target_include_directories(vf_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
find_package(ZLIB REQUIRED)
target_link_libraries(vf_io PUBLIC ZLIB::ZLIB)

# ---- Main executable ----
add_executable(visionforge_app apps/visionforge/main.cpp)
target_link_libraries(visionforge_app PRIVATE visionforge vf_io)

# ---- OpenMP (parallel render) ----
if (VISIONFORGE_OMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling multithreaded sampler")
    target_link_libraries(visionforge_app PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(visionforge_app PUBLIC VF_USE_OMP=1)
  else()
    message(STATUS "OpenMP NOT found: building single-threaded")
  endif()
endif()

# ---- Optimizations ----
if (MSVC)
  target_compile_options(vf_io           PRIVATE /O2 /DNDEBUG)
  target_compile_options(visionforge_app PRIVATE /O2 /DNDEBUG)
  # Optional: enable AVX2 if your CPUs support it
  # target_compile_options(vf_io           PRIVATE /arch:AVX2)
  # target_compile_options(visionforge_app PRIVATE /arch:AVX2)
else()
  target_compile_options(vf_io           PRIVATE -O3 -march=native -DNDEBUG -ffast-math)
  target_compile_options(visionforge_app PRIVATE -O3 -march=native -DNDEBUG -ffast-math)
endif()

# ---- (Optional) IPO/LTO ----
if (VISIONFORGE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
  if (ipo_ok)
    set_property(TARGET visionforge_app PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET vf_io            PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO/LTO not supported: ${ipo_msg}")
  endif()
endif()

# ---- Output name ----
set_target_properties(visionforge_app PROPERTIES OUTPUT_NAME "visionforge")

# ---- Install + simple packaging ----
install(TARGETS visionforge_app RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

set(CPACK_PACKAGE_NAME "visionforge")
set(CPACK_PACKAGE_VENDOR "VisionForge")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)
