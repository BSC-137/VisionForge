#pragma once
#include "ray.hpp"
#include "hittable.hpp"
#include <memory>  // for enable_shared_from_this
#include <algorithm>

class Material : public std::enable_shared_from_this<Material> {
public:
    Vec3 base_color;
    double roughness;
    double metallic;

    Material(const Vec3& base = Vec3(0.8, 0.8, 0.8), double rough = 0.5, double metal = 0.0)
        : base_color(base),
          roughness(std::clamp(rough, 0.0, 1.0)),
          metallic(std::clamp(metal, 0.0, 1.0)) {}

    // BRDF sampling
    virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                         Vec3& attenuation, Ray& scattered) const = 0;

    // Emission (radiance, W·sr^-1·m^-2); default = black
    virtual Vec3 emitted(const HitRecord&) const { return Vec3(0,0,0); }

    virtual bool uses_microfacet() const { return false; }
    virtual bool is_emissive() const { return false; }

    virtual ~Material() = default;
};
