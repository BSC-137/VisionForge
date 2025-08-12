#pragma once
#include <memory>
#include <cmath>
#include "hittable.hpp"

class XZRect : public Hittable {
public:
    double x0, x1, z0, z1, k; // plane y = k
    std::shared_ptr<Material> mat;
    static constexpr double THICK = 1e-4;

    XZRect(double _x0, double _x1, double _z0, double _z1, double _k,
           std::shared_ptr<Material> m)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat(std::move(m)) {}

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        double denom = r.direction.y;
        if (std::fabs(denom) < 1e-8) return false;
        double t = (k - r.origin.y) / denom;
        if (t < t_min || t > t_max) return false;

        double x = r.origin.x + t * r.direction.x;
        double z = r.origin.z + t * r.direction.z;
        if (x < x0 || x > x1 || z < z0 || z > z1) return false;

        rec.t = t;
        rec.point = r.at(t);
        rec.set_face_normal(r, Vec3(0,1,0));
        rec.mat = mat;
        rec.object_id = this->id;          // NEW
        return true;
    }

    double area() const { return (x1 - x0) * (z1 - z0); }
    Vec3   sample_point() const {
        return Vec3(random_double(x0, x1), k, random_double(z0, z1));
    }
    Vec3   light_normal() const { return Vec3(0,-1,0); }

    bool bounding_box(AABB& out_box) const override {
        out_box = AABB(Vec3(x0, k - THICK, z0), Vec3(x1, k + THICK, z1));
        return true;
    }
};
