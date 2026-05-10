#pragma once
#include <memory>
#include <cmath>
#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/aabb.hpp"

// ---- Slope rotation from terrain normal (Rodrigues) ----

struct SlopeRotation {
    double m[3][3];
};

// Computes the rotation matrix that maps Up=(0,1,0) to the given terrain
// normal N.  Uses the closed-form Rodrigues derivation with axis =
// normalize(cross(Up, N)).  Fully inlined; no trig calls beyond what the
// compiler can constant-fold.
static inline SlopeRotation slope_rotation_from_normal(const Vec3& N) {
    SlopeRotation rot;
    const double Nx = N.x, Ny = N.y, Nz = N.z;
    const double s2 = Nx * Nx + Nz * Nz;

    if (s2 < 1e-12) {
        const double sign = (Ny >= 0.0) ? 1.0 : -1.0;
        rot.m[0][0] = 1; rot.m[0][1] = 0; rot.m[0][2] = 0;
        rot.m[1][0] = 0; rot.m[1][1] = sign; rot.m[1][2] = 0;
        rot.m[2][0] = 0; rot.m[2][1] = 0; rot.m[2][2] = 1;
        return rot;
    }

    // t = (1 - cos theta) / sin^2 theta = 1 / (1 + Ny)
    const double t = (1.0 - Ny) / s2;

    rot.m[0][0] = 1.0 - t * Nx * Nx;
    rot.m[0][1] = Nx;
    rot.m[0][2] = -t * Nx * Nz;

    rot.m[1][0] = -Nx;
    rot.m[1][1] = Ny;
    rot.m[1][2] = -Nz;

    rot.m[2][0] = -t * Nx * Nz;
    rot.m[2][1] = Nz;
    rot.m[2][2] = 1.0 - t * Nz * Nz;

    return rot;
}

// ---- Gravity snap ----

static inline double snap_y(double terrain_height, double bbox_height, double sinking_depth = 0.0) {
    return terrain_height + bbox_height * 0.5 - sinking_depth;
}

// ---- SlopeAlign Hittable wrapper ----
// Applies a general 3×3 rotation (from slope_rotation_from_normal) to a
// child Hittable.  Computed once per object at construction time; the
// per-ray cost is two matrix-vector multiplies (same as RotateY).

class SlopeAlign : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double R[3][3];    // local → world
    double Ri[3][3];   // world → local  (transpose of R since R is orthogonal)
    AABB   box;

    SlopeAlign(std::shared_ptr<Hittable> p, const Vec3& terrain_normal)
        : ptr(std::move(p))
    {
        set_from_normal(terrain_normal);
    }

    void set_from_normal(const Vec3& terrain_normal) {
        SlopeRotation rot = slope_rotation_from_normal(terrain_normal);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                R[i][j]  = rot.m[i][j];
                Ri[j][i] = rot.m[i][j];
            }
        recompute_aabb();
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        // Rotate ray from world space into local (unrotated) space
        const Vec3& ro = r.origin;
        const Vec3& rd = r.direction;
        Vec3 o(Ri[0][0]*ro.x + Ri[0][1]*ro.y + Ri[0][2]*ro.z,
               Ri[1][0]*ro.x + Ri[1][1]*ro.y + Ri[1][2]*ro.z,
               Ri[2][0]*ro.x + Ri[2][1]*ro.y + Ri[2][2]*ro.z);
        Vec3 d(Ri[0][0]*rd.x + Ri[0][1]*rd.y + Ri[0][2]*rd.z,
               Ri[1][0]*rd.x + Ri[1][1]*rd.y + Ri[1][2]*rd.z,
               Ri[2][0]*rd.x + Ri[2][1]*rd.y + Ri[2][2]*rd.z);

        Ray rr(o, d, r.time);
        if (!ptr->hit(rr, tmin, tmax, rec)) return false;

        // Rotate hit point and normal back to world space
        Vec3 pt = rec.point;
        rec.point.x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
        rec.point.y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
        rec.point.z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;

        Vec3 n = rec.normal;
        rec.normal.x = R[0][0]*n.x + R[0][1]*n.y + R[0][2]*n.z;
        rec.normal.y = R[1][0]*n.x + R[1][1]*n.y + R[1][2]*n.z;
        rec.normal.z = R[2][0]*n.x + R[2][1]*n.y + R[2][2]*n.z;

        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = box;
        return true;
    }

private:
    void recompute_aabb() {
        AABB b;
        if (!ptr->bounding_box(b)) return;
        Vec3 minv(1e9, 1e9, 1e9), maxv(-1e9, -1e9, -1e9);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const double x = i ? b.max().x : b.min().x;
                    const double y = j ? b.max().y : b.min().y;
                    const double z = k ? b.max().z : b.min().z;
                    Vec3 p(R[0][0]*x + R[0][1]*y + R[0][2]*z,
                           R[1][0]*x + R[1][1]*y + R[1][2]*z,
                           R[2][0]*x + R[2][1]*y + R[2][2]*z);
                    minv = min_vec(minv, p);
                    maxv = max_vec(maxv, p);
                }
        box = AABB(minv, maxv);
    }
};
