#pragma once
#include "material.hpp"

// Lambertian emitter: user provides exitance M (W·m^-2) and color tint.
// Emitted radiance Le = M * tint / π  (physically correct for diffuse emitters)
class DiffuseLight : public Material {
public:
    Vec3 tint;      // spectral tint (0..1 each channel)
    double exitance; // radiant exitance M (W·m^-2)

    DiffuseLight(const Vec3& tint_, double exitance_)
        : tint(tint_), exitance(exitance_) {}

    bool scatter(const Ray&, const HitRecord&, Vec3&, Ray&) const override {
        // Lights don't scatter in this simple model
        return false;
    }

    Vec3 emitted(const HitRecord&) const override {
        const double inv_pi = 1.0 / 3.14159265358979323846;
        return exitance * inv_pi * tint; // Le (radiance)
    }
};
