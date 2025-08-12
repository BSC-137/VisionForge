#pragma once
#include <memory>
#include "hittable.hpp"

class Sphere : public Hittable {
public:
    Vec3 center;
    double radius;
    std::shared_ptr<Material> mat;

    Sphere(const Vec3& c, double r, std::shared_ptr<Material> m)
        : center(c), radius(r), mat(std::move(m)) {}

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 oc = r.origin - center;
        double a = dot(r.direction, r.direction);
        double half_b = dot(oc, r.direction);
        double c = dot(oc, oc) - radius * radius;
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;

        double sqrt_d = std::sqrt(discriminant);

        double root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.point = r.at(rec.t);
        Vec3 outward_normal = (rec.point - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        // NEW: carry instance id into the hit record
        rec.object_id = this->id;

        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        Vec3 r(radius, radius, radius);
        out_box = AABB(center - r, center + r);
        return true;
    }
};
