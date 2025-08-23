#pragma once
#include <memory>
#include <cmath>
#include "hittable.hpp"

class XYRect : public Hittable {
public:
    double x0, x1, y0, y1, k; // plane z = k
    std::shared_ptr<Material> mat;
    static constexpr double THICK = 1e-4;

    XYRect(double _x0, double _x1, double _y0, double _y1, double _k,
           std::shared_ptr<Material> m)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat(std::move(m)) {}

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (std::fabs(r.direction.z) < 1e-8) return false;
        double t = (k - r.origin.z) / r.direction.z;
        if (t < t_min || t > t_max) return false;

        double x = r.origin.x + t * r.direction.x;
        double y = r.origin.y + t * r.direction.y;
        if (x < x0 || x > x1 || y < y0 || y > y1) return false;

        rec.t = t;
        rec.point = r.at(t);
        rec.set_face_normal(r, Vec3(0,0,1));
        rec.mat = mat;
        rec.hit_object = this;
        return true;
    }

    double area() const { return (x1 - x0) * (y1 - y0); }

    bool bounding_box(AABB& out_box) const override {
        out_box = AABB(Vec3(x0, y0, k - THICK), Vec3(x1, y1, k + THICK));
        return true;
    }
};
