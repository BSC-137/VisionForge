#pragma once
#include <memory>
#include "ray.hpp"
#include "aabb.hpp"

class Material; // forward declaration

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t;
    bool front_face;
    std::shared_ptr<Material> mat;

    // NEW: carry the instance/object id to AOVs/masks
    int object_id = 0;

    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        // assumes Ray exposes public 'direction'
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    // NEW: per-instance id; set this when you create the object
    int id = 0;

    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
    virtual bool bounding_box(AABB& out_box) const = 0;
};
