#pragma once
#include "material.hpp"
#include "vec3.hpp"

class Lambertian : public Material {
public:
    Vec3 albedo;
    explicit Lambertian(const Vec3& a) : albedo(a) {}

    bool scatter(const Ray&, const HitRecord& rec,
                 Vec3& attenuation, Ray& scattered) const override {
        Vec3 scatter_dir = rec.normal + random_unit_vector();
        if (near_zero(scatter_dir)) scatter_dir = rec.normal;
        scattered = Ray(rec.point, scatter_dir);
        attenuation = albedo; // cosine-weighted diffuse => weight collapses to albedo
        return true;
    }
};
