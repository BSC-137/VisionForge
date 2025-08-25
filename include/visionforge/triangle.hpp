#pragma once
#include <memory>
#include <cmath>
#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/material.hpp"
#include "visionforge/aabb.hpp"

// Single triangle with per-vertex normals (for smooth terrain).
class Triangle : public Hittable {
public:
    Vec3 p0, p1, p2;           // vertices
    Vec3 n0, n1, n2;           // vertex normals (assumed normalized)
    std::shared_ptr<Material> mat;
    AABB box;

    Triangle() = default;

    Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
             const Vec3& na, const Vec3& nb, const Vec3& nc,
             std::shared_ptr<Material> m)
    : p0(a), p1(b), p2(c),
      n0(normalize(na)), n1(normalize(nb)), n2(normalize(nc)),
      mat(std::move(m))
    {
        const double eps = 1e-6;
        Vec3 mn(std::fmin(p0.x, std::fmin(p1.x, p2.x)),
                std::fmin(p0.y, std::fmin(p1.y, p2.y)),
                std::fmin(p0.z, std::fmin(p1.z, p2.z)));
        Vec3 mx(std::fmax(p0.x, std::fmax(p1.x, p2.x)),
                std::fmax(p0.y, std::fmax(p1.y, p2.y)),
                std::fmax(p0.z, std::fmax(p1.z, p2.z)));
        box = AABB(mn - Vec3(eps,eps,eps), mx + Vec3(eps,eps,eps));
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        // Möller–Trumbore
        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const Vec3 pvec = cross(r.direction, e2);
        const double det = dot(e1, pvec);
        if (std::fabs(det) < 1e-10) return false;
        const double invDet = 1.0 / det;

        const Vec3 tvec = r.origin - p0;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) return false;

        const Vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction, qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * invDet;
        if (t < t_min || t > t_max) return false;

        rec.t = t;
        rec.point = r.at(t);

        const double w = 1.0 - u - v;
        Vec3 n = normalize(w * n0 + u * n1 + v * n2);
        if (dot(n, r.direction) > 0) n = -n;   // face toward ray
        rec.normal = n;
        rec.mat = mat;

        // This triangle isn't tagged; keep hit_object to allow outer wrappers (e.g., Tag) to set it.
        rec.hit_object = this;
        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = box;
        return true;
    }
};
