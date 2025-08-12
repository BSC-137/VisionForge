#pragma once
#include <memory>
#include "hittable.hpp"

class MovingSphere : public Hittable {
public:
    Vec3 center0, center1;
    double time0, time1;
    double radius;
    std::shared_ptr<Material> mat;

    MovingSphere(const Vec3& c0, const Vec3& c1,
                 double t0, double t1,
                 double r, std::shared_ptr<Material> m)
        : center0(c0), center1(c1), time0(t0), time1(t1), radius(r), mat(std::move(m)) {}

    Vec3 center(double time) const {
        double alpha = (time - time0) / (time1 - time0);
        return center0 + alpha * (center1 - center0);
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 c = center(r.time);
        Vec3 oc = r.origin - c;
        double a = dot(r.direction, r.direction);
        double half_b = dot(oc, r.direction);
        double cc = dot(oc, oc) - radius*radius;
        double discriminant = half_b*half_b - a*cc;
        if (discriminant < 0) return false;

        double sqrt_d = std::sqrt(discriminant);

        double root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.point = r.at(rec.t);
        Vec3 outward_normal = (rec.point - c) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        return true;
    }
};
