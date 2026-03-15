#pragma once
#include <memory>
#include <cmath>
#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/aabb.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// ---------------- Translate ----------------
class Translate : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    Vec3 offset;

    Translate(std::shared_ptr<Hittable> p, const Vec3& d)
    : ptr(std::move(p)), offset(d) {}

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Ray moved(r.origin - offset, r.direction, r.time);
        if (!ptr->hit(moved, t_min, t_max, rec)) return false;
        rec.point += offset;
        // keep rec.hit_object as set by child (important for Tag)
        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        AABB b;
        if (!ptr->bounding_box(b)) return false;
        out_box = AABB(b.min() + offset, b.max() + offset);
        return true;
    }
};

// ---------------- RotateX ----------------
class RotateX : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double sin_t, cos_t;
    AABB box;

    explicit RotateX(std::shared_ptr<Hittable> p, double angle_deg)
    : ptr(std::move(p)) {
        double rad = angle_deg * PI / 180.0;
        sin_t = std::sin(rad);
        cos_t = std::cos(rad);

        AABB b;
        if (ptr->bounding_box(b)) {
            Vec3 minv( 1e9, 1e9, 1e9), maxv(-1e9,-1e9,-1e9);
            for (int i=0;i<2;i++) for (int j=0;j<2;j++) for (int k=0;k<2;k++) {
                double x = i ? b.max().x : b.min().x;
                double y = j ? b.max().y : b.min().y;
                double z = k ? b.max().z : b.min().z;
                // rotate box corners around X
                double ny =  cos_t*y - sin_t*z;
                double nz =  sin_t*y + cos_t*z;
                Vec3 p(x, ny, nz);
                minv = Vec3(std::min(minv.x,p.x), std::min(minv.y,p.y), std::min(minv.z,p.z));
                maxv = Vec3(std::max(maxv.x,p.x), std::max(maxv.y,p.y), std::max(maxv.z,p.z));
            }
            box = AABB(minv, maxv);
        }
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        Vec3 o = r.origin, d = r.direction;
        // rotate ray into object space
        o.y =  cos_t*r.origin.y + sin_t*r.origin.z;
        o.z = -sin_t*r.origin.y + cos_t*r.origin.z;
        d.y =  cos_t*r.direction.y + sin_t*r.direction.z;
        d.z = -sin_t*r.direction.y + cos_t*r.direction.z;

        Ray rr(o, d, r.time);
        if (!ptr->hit(rr, tmin, tmax, rec)) return false;

        // rotate hit back to world space
        double y =  cos_t*rec.point.y - sin_t*rec.point.z;
        double z =  sin_t*rec.point.y + cos_t*rec.point.z;
        rec.point.y = y; rec.point.z = z;

        double ny =  cos_t*rec.normal.y - sin_t*rec.normal.z;
        double nz =  sin_t*rec.normal.y + cos_t*rec.normal.z;
        rec.normal.y = ny; rec.normal.z = nz;
        return true;
    }

    bool bounding_box(AABB& out_box) const override { out_box = box; return true; }
};

// ---------------- RotateZ ----------------
class RotateZ : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double sin_t, cos_t;
    AABB box;

    explicit RotateZ(std::shared_ptr<Hittable> p, double angle_deg)
    : ptr(std::move(p)) {
        double rad = angle_deg * PI / 180.0;
        sin_t = std::sin(rad);
        cos_t = std::cos(rad);

        AABB b;
        if (ptr->bounding_box(b)) {
            Vec3 minv( 1e9, 1e9, 1e9), maxv(-1e9,-1e9,-1e9);
            for (int i=0;i<2;i++) for (int j=0;j<2;j++) for (int k=0;k<2;k++) {
                double x = i ? b.max().x : b.min().x;
                double y = j ? b.max().y : b.min().y;
                double z = k ? b.max().z : b.min().z;
                // rotate box corners around Z
                double nx =  cos_t*x - sin_t*y;
                double ny =  sin_t*x + cos_t*y;
                Vec3 p(nx, ny, z);
                minv = Vec3(std::min(minv.x,p.x), std::min(minv.y,p.y), std::min(minv.z,p.z));
                maxv = Vec3(std::max(maxv.x,p.x), std::max(maxv.y,p.y), std::max(maxv.z,p.z));
            }
            box = AABB(minv, maxv);
        }
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        Vec3 o = r.origin, d = r.direction;
        // rotate ray into object space
        o.x =  cos_t*r.origin.x + sin_t*r.origin.y;
        o.y = -sin_t*r.origin.x + cos_t*r.origin.y;
        d.x =  cos_t*r.direction.x + sin_t*r.direction.y;
        d.y = -sin_t*r.direction.x + cos_t*r.direction.y;

        Ray rr(o, d, r.time);
        if (!ptr->hit(rr, tmin, tmax, rec)) return false;

        // rotate hit back to world space
        double x =  cos_t*rec.point.x - sin_t*rec.point.y;
        double y =  sin_t*rec.point.x + cos_t*rec.point.y;
        rec.point.x = x; rec.point.y = y;

        double nx =  cos_t*rec.normal.x - sin_t*rec.normal.y;
        double ny =  sin_t*rec.normal.x + cos_t*rec.normal.y;
        rec.normal.x = nx; rec.normal.y = ny;
        return true;
    }

    bool bounding_box(AABB& out_box) const override { out_box = box; return true; }
};

// ---------------- RotateY ----------------
class RotateY : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double sin_t, cos_t;
    AABB box;

    explicit RotateY(std::shared_ptr<Hittable> p, double angle_deg)
    : ptr(std::move(p)) {
        double rad = angle_deg * PI / 180.0;
        sin_t = std::sin(rad);
        cos_t = std::cos(rad);

        AABB b;
        if (ptr->bounding_box(b)) {
            Vec3 minv( 1e9, 1e9, 1e9), maxv(-1e9,-1e9,-1e9);
            for (int i=0;i<2;i++) for (int j=0;j<2;j++) for (int k=0;k<2;k++) {
                double x = i ? b.max().x : b.min().x;
                double y = j ? b.max().y : b.min().y;
                double z = k ? b.max().z : b.min().z;
                double nx =  cos_t*x + sin_t*z;
                double nz = -sin_t*x + cos_t*z;
                Vec3 p(nx, y, nz);
                minv = Vec3(std::min(minv.x,p.x), std::min(minv.y,p.y), std::min(minv.z,p.z));
                maxv = Vec3(std::max(maxv.x,p.x), std::max(maxv.y,p.y), std::max(maxv.z,p.z));
            }
            box = AABB(minv, maxv);
        }
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        Vec3 o = r.origin, d = r.direction;
        o.x =  cos_t*r.origin.x - sin_t*r.origin.z;
        o.z =  sin_t*r.origin.x + cos_t*r.origin.z;
        d.x =  cos_t*r.direction.x - sin_t*r.direction.z;
        d.z =  sin_t*r.direction.x + cos_t*r.direction.z;

        Ray rr(o, d, r.time);
        if (!ptr->hit(rr, tmin, tmax, rec)) return false;

        double x = cos_t*rec.point.x + sin_t*rec.point.z;
        double z =-sin_t*rec.point.x + cos_t*rec.point.z;
        rec.point.x = x; rec.point.z = z;

        double nx = cos_t*rec.normal.x + sin_t*rec.normal.z;
        double nz =-sin_t*rec.normal.x + cos_t*rec.normal.z;
        rec.normal.x = nx; rec.normal.z = nz;
        return true;
    }

    bool bounding_box(AABB& out_box) const override { out_box = box; return true; }
};

// ---------------- Scale ----------------
class Scale : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    Vec3 scale;
    Vec3 inv_scale;

    Scale(std::shared_ptr<Hittable> p, const Vec3& s)
    : ptr(std::move(p)), scale(s) {
        inv_scale = Vec3(
            s.x == 0.0 ? 0.0 : 1.0 / s.x,
            s.y == 0.0 ? 0.0 : 1.0 / s.y,
            s.z == 0.0 ? 0.0 : 1.0 / s.z
        );
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 scaled_origin(r.origin.x * inv_scale.x, r.origin.y * inv_scale.y, r.origin.z * inv_scale.z);
        Vec3 scaled_dir(r.direction.x * inv_scale.x, r.direction.y * inv_scale.y, r.direction.z * inv_scale.z);
        
        Ray scaled_ray(scaled_origin, scaled_dir, r.time);
        
        if (!ptr->hit(scaled_ray, t_min, t_max, rec)) return false;
        
        rec.point = Vec3(rec.point.x * scale.x, rec.point.y * scale.y, rec.point.z * scale.z);
        
        // Normal transform for non-uniform scale: N = normalize(inv_scale * inv_scale * N) or similar
        // A simpler exact way is Normal = normalize( vec3(n.x*inv.x, n.y*inv.y, n.z*inv.z) );
        rec.normal = normalize(Vec3(
            rec.normal.x * inv_scale.x,
            rec.normal.y * inv_scale.y,
            rec.normal.z * inv_scale.z
        ));
        
        return true;
    }

    bool bounding_box(AABB& out_box) const override {
        AABB b;
        if (!ptr->bounding_box(b)) return false;
        out_box = AABB(
            Vec3(b.min().x * scale.x, b.min().y * scale.y, b.min().z * scale.z),
            Vec3(b.max().x * scale.x, b.max().y * scale.y, b.max().z * scale.z)
        );
        return true;
    }
};
