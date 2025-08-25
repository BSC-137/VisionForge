#pragma once
#include <memory>
#include <cmath>
#include "hittable.hpp"

class YZRect : public Hittable {
public:
    double y0, y1, z0, z1, k; // plane x = k
    std::shared_ptr<Material> mat;
    static constexpr double THICK = 1e-4;

    YZRect(double _y0, double _y1, double _z0, double _z1, double _k,
           std::shared_ptr<Material> m)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat(std::move(m)) {}

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (std::fabs(r.direction.x) < 1e-8) return false;
        double t = (k - r.origin.x) / r.direction.x;
        if (t < t_min || t > t_max) return false;

        double y = r.origin.y + t * r.direction.y;
        double z = r.origin.z + t * r.direction.z;
        if (y < y0 || y > y1 || z < z0 || z > z1) return false;

        rec.t = t;
        rec.point = r.at(t);
        rec.set_face_normal(r, Vec3(1,0,0));
        rec.mat = mat;
        rec.hit_object = this;
        return true;
    }

    double area() const { return (y1 - y0) * (z1 - z0); }

    // For use as an emissive area light (wall at x=k shining +X)
    Vec3 light_normal() const { return Vec3(1,0,0); } // +X
    Vec3 sample_point() const {
        double y = random_double(y0, y1);
        double z = random_double(z0, z1);
        return Vec3(k, y, z);
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = AABB(Vec3(k - THICK, y0, z0), Vec3(k + THICK, y1, z1));
        return true;
    }
};
