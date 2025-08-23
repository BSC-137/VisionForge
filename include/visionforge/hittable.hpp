#pragma once
#include <memory>
#include <cstdint>          // <-- add this
#include "ray.hpp"
#include "aabb.hpp"

class Material;             // fwd
class Hittable;             // <-- add this forward declaration

// ---- Object tagging for dataset mode ----
struct ObjectInfo {
    uint32_t instance_id = 0;   // 0 = background
    uint32_t class_id    = 0;   // e.g., 1 = cube
    const char* label    = "";  // readable name
};

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t = 0.0;
    bool front_face = true;
    std::shared_ptr<Material> mat;

    // Which object was hit (shape or Tag wrapper)
    const Hittable* hit_object = nullptr;   // <-- use this, not an integer id

    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    ObjectInfo obj;  // per-object IDs/label live here
    virtual bool hit(const Ray&, double, double, HitRecord&) const = 0;
    virtual bool bounding_box(AABB& out_box) const = 0;
    virtual ~Hittable() = default;
};
