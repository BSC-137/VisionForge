#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/material.hpp"
#include "visionforge/aabb.hpp"

struct MeshData {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
};

class MeshTriangle : public Hittable {
public:
    std::shared_ptr<const MeshData> data;
    uint32_t i0, i1, i2;
    uint32_t n0, n1, n2;
    bool has_normals;
    std::shared_ptr<Material> mat;
    AABB box;

    MeshTriangle(std::shared_ptr<const MeshData> d,
                 uint32_t pi0, uint32_t pi1, uint32_t pi2,
                 uint32_t ni0, uint32_t ni1, uint32_t ni2,
                 bool has_nrm,
                 std::shared_ptr<Material> m)
        : data(std::move(d)),
          i0(pi0), i1(pi1), i2(pi2),
          n0(ni0), n1(ni1), n2(ni2),
          has_normals(has_nrm),
          mat(std::move(m))
    {
        const Vec3& p0 = data->positions[i0];
        const Vec3& p1 = data->positions[i1];
        const Vec3& p2 = data->positions[i2];
        constexpr double eps = 1e-6;
        Vec3 mn(std::fmin(p0.x, std::fmin(p1.x, p2.x)),
                std::fmin(p0.y, std::fmin(p1.y, p2.y)),
                std::fmin(p0.z, std::fmin(p1.z, p2.z)));
        Vec3 mx(std::fmax(p0.x, std::fmax(p1.x, p2.x)),
                std::fmax(p0.y, std::fmax(p1.y, p2.y)),
                std::fmax(p0.z, std::fmax(p1.z, p2.z)));
        box = AABB(mn - Vec3(eps, eps, eps), mx + Vec3(eps, eps, eps));
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        const Vec3& p0 = data->positions[i0];
        const Vec3& p1 = data->positions[i1];
        const Vec3& p2 = data->positions[i2];

        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const Vec3 pvec = cross(r.direction, e2);
        const double det = dot(e1, pvec);
        if (std::fabs(det) < 1e-10) return false;
        const double inv_det = 1.0 / det;

        const Vec3 tvec = r.origin - p0;
        const double u = dot(tvec, pvec) * inv_det;
        if (u < 0.0 || u > 1.0) return false;

        const Vec3 qvec = cross(tvec, e1);
        const double v = dot(r.direction, qvec) * inv_det;
        if (v < 0.0 || u + v > 1.0) return false;

        const double t = dot(e2, qvec) * inv_det;
        if (t < t_min || t > t_max) return false;

        rec.t = t;
        rec.point = r.at(t);

        Vec3 n;
        if (has_normals && n0 < data->normals.size()) {
            const double w = 1.0 - u - v;
            n = normalize(w * data->normals[n0] + u * data->normals[n1] + v * data->normals[n2]);
        } else {
            n = normalize(cross(e1, e2));
        }

        rec.set_face_normal(r, n);
        rec.mat = mat;
        rec.hit_object = this;
        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = box;
        return true;
    }
};
