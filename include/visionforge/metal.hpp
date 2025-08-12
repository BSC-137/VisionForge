#pragma once
#include "material.hpp"
#include "vec3.hpp"

class Metal : public Material {
public:
    Vec3 albedo;
    double fuzz;
    Metal(const Vec3& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(const Ray& r_in, const HitRecord& rec,
                 Vec3& attenuation, Ray& scattered) const override {
        Vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
        scattered = Ray(rec.point, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return dot(scattered.direction, rec.normal) > 0;
    }
};
