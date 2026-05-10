#pragma once
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>
#include "visionforge/vec3.hpp"
#include "visionforge/image_texture.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

class HDRSky {
public:
    HDRSky() = default;

    bool load(const std::string& path, double intensity = 1.0) {
        intensity_ = intensity;
        return tex_.load(path);
    }

    bool valid() const { return tex_.valid(); }
    void set_intensity(double i) { intensity_ = i; }

    inline Vec3 eval(const Vec3& dir) const {
        Vec3 d = normalize(dir);
        double u = 0.5 + std::atan2(d.z, d.x) / (2.0 * PI);
        double v = 0.5 - std::asin(std::clamp(d.y, -1.0, 1.0)) / PI;
        return intensity_ * tex_.sample(u, v);
    }

private:
    ImageTexture tex_;
    double intensity_ = 1.0;
};
