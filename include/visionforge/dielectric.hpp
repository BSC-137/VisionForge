#pragma once
#include "material.hpp"
#include "vec3.hpp"
#include <cmath>

class Dielectric : public Material {
public:
    double ir; // index of refraction (e.g., 1.5 for glass)
    explicit Dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    static double reflectance(double cos, double ref_idx) {
        // Schlick approximation (average reflectance)
        double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cos, 5.0);
    }

    bool scatter(const Ray& r_in, const HitRecord& rec,
                 Vec3& attenuation, Ray& scattered) const override {
        attenuation = Vec3(1.0, 1.0, 1.0); // clear glass
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        Vec3 unit_dir = normalize(r_in.direction);
        double cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
            direction = reflect(unit_dir, rec.normal);
        } else {
            direction = refract(unit_dir, rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.point, direction);
        return true;
    }
};
