cmake_minimum_required(VERSION 3.16)
project(VisionForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core library for headers
add_library(visionforge INTERFACE)
target_include_directories(visionforge INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# TinyEXR I/O library
add_library(vf_io STATIC
    src/io/exr_writer.cpp
)
target_include_directories(vf_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
find_package(ZLIB REQUIRED)
target_link_libraries(vf_io PUBLIC ZLIB::ZLIB)

# Main executable
add_executable(visionforge_app apps/visionforge/main.cpp)
target_link_libraries(visionforge_app PRIVATE visionforge vf_io)
set_target_properties(visionforge_app PROPERTIES OUTPUT_NAME "visionforge")
