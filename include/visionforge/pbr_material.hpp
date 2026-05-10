#pragma once

#include <algorithm>
#include <cmath>

#include "visionforge/material.hpp"
#include "visionforge/vec3.hpp"

class ImageTexture;

namespace vf_pbr {

inline double schlick5(double x) {
    const double x2 = x * x;
    return x2 * x2 * x;
}

inline Vec3 fresnel_schlick(double cos_theta, const Vec3& F0) {
    const double t = schlick5(std::clamp(1.0 - cos_theta, 0.0, 1.0));
    return F0 + (Vec3(1.0, 1.0, 1.0) - F0) * t;
}

inline void build_onb(const Vec3& n, Vec3& t, Vec3& b) {
    const Vec3 a = (std::fabs(n.y) < 0.999) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    t = normalize(cross(a, n));
    b = cross(n, t);
}

inline Vec3 local_to_world(const Vec3& local, const Vec3& t, const Vec3& b, const Vec3& n) {
    return t * local.x + b * local.y + n * local.z;
}

inline Vec3 sample_cosine_hemisphere(const Vec3& n, const Vec3& t, const Vec3& b) {
    const double u1 = random_double();
    const double u2 = random_double();
    const double r = std::sqrt(u1);
    const double phi = 2.0 * PI * u2;
    return normalize(local_to_world(Vec3(r * std::cos(phi), r * std::sin(phi),
                                         std::sqrt(std::max(0.0, 1.0 - u1))), t, b, n));
}

inline Vec3 sample_ggx_half_vector(const Vec3& t, const Vec3& b, const Vec3& n, double a2) {
    const double u1 = random_double();
    const double u2 = random_double();
    const double phi = 2.0 * PI * u1;
    const double cos_theta = std::sqrt((1.0 - u2) / (1.0 + (a2 - 1.0) * u2));
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    return normalize(local_to_world(Vec3(sin_theta * std::cos(phi),
                                         sin_theta * std::sin(phi),
                                         cos_theta), t, b, n));
}

} // namespace vf_pbr

class PBRMaterial : public Material {
public:
    std::shared_ptr<ImageTexture> albedo_map;
    std::shared_ptr<ImageTexture> normal_map;
    std::shared_ptr<ImageTexture> roughness_map;
    std::shared_ptr<ImageTexture> metallic_map;

    explicit PBRMaterial(const Vec3& color = Vec3(0.8, 0.8, 0.8), double rough = 0.5, double metal = 0.0)
        : Material(color, rough, metal) {}

    void set_parameters(const Vec3& color, double rough, double metal) {
        base_color = color;
        roughness = std::clamp(rough, 0.0, 1.0);
        metallic = std::clamp(metal, 0.0, 1.0);
    }

    Mat3 get_tbn_matrix(const HitRecord& rec) const {
        Vec3 t, b;
        vf_pbr::build_onb(rec.normal, t, b);
        return Mat3(t, b, rec.normal);
    }

    bool uses_microfacet() const override { return true; }

    bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
};
