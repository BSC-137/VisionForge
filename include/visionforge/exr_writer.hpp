#pragma once
#include <vector>
#include "visionforge/vec3.hpp"

namespace vf {
bool write_rgb_exr(const char* path, int w, int h, const std::vector<Vec3>& rgb);
bool write_float_exr(const char* path, int w, int h, const std::vector<float>& img, const char* channel="Y");
}
