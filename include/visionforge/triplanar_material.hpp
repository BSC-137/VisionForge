#pragma once
#include <memory>
#include <cmath>
#include "visionforge/material.hpp"
#include "visionforge/vec3.hpp"
#include "visionforge/image_texture.hpp"

class TriplanarMaterial : public Material {
public:
    std::shared_ptr<ImageTexture> tex;
    double tex_scale;

    TriplanarMaterial(std::shared_ptr<ImageTexture> t, double scale = 2.0)
        : Material(Vec3(0.78, 0.72, 0.58), 1.0, 0.0),
          tex(std::move(t)), tex_scale(scale) {}

    bool scatter(const Ray&, const HitRecord& rec,
                 Vec3& attenuation, Ray& scattered) const override {
        Vec3 n = rec.normal;
        double wx = std::fabs(n.x);
        double wy = std::fabs(n.y);
        double wz = std::fabs(n.z);
        wx = wx * wx * wx * wx;
        wy = wy * wy * wy * wy;
        wz = wz * wz * wz * wz;
        double wsum = wx + wy + wz;
        if (wsum < 1e-12) wsum = 1.0;
        wx /= wsum; wy /= wsum; wz /= wsum;

        Vec3 p = rec.point * tex_scale;
        Vec3 cx = tex->sample(p.y, p.z);
        Vec3 cy = tex->sample(p.x, p.z);
        Vec3 cz = tex->sample(p.x, p.y);

        attenuation = wx * cx + wy * cy + wz * cz;

        Vec3 scatter_dir = rec.normal + random_unit_vector();
        if (near_zero(scatter_dir)) scatter_dir = rec.normal;
        scattered = Ray(rec.point, scatter_dir);
        return true;
    }
};
