#pragma once
#include "ray.hpp"
#include "hittable.hpp"

class Material {
public:
    // BRDF sampling
    virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                         Vec3& attenuation, Ray& scattered) const = 0;

    // Emission (radiance, W·sr^-1·m^-2); default = black
    virtual Vec3 emitted(const HitRecord& rec) const { return Vec3(0,0,0); }

    virtual ~Material() = default;
};
