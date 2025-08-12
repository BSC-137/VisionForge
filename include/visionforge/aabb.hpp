#pragma once
#include "ray.hpp"

// Axis-aligned bounding box (slab test)
struct AABB {
    Vec3 minimum;
    Vec3 maximum;

    AABB() : minimum(), maximum() {}
    AABB(const Vec3& a, const Vec3& b) : minimum(a), maximum(b) {}

    const Vec3& min() const { return minimum; }
    const Vec3& max() const { return maximum; }

    bool hit(const Ray& r, double t_min, double t_max) const {
        // X
        double invDx = 1.0 / r.direction.x;
        double tx0 = (min().x - r.origin.x) * invDx;
        double tx1 = (max().x - r.origin.x) * invDx;
        if (invDx < 0.0) std::swap(tx0, tx1);
        t_min = tx0 > t_min ? tx0 : t_min;
        t_max = tx1 < t_max ? tx1 : t_max;
        if (t_max <= t_min) return false;

        // Y
        double invDy = 1.0 / r.direction.y;
        double ty0 = (min().y - r.origin.y) * invDy;
        double ty1 = (max().y - r.origin.y) * invDy;
        if (invDy < 0.0) std::swap(ty0, ty1);
        t_min = ty0 > t_min ? ty0 : t_min;
        t_max = ty1 < t_max ? ty1 : t_max;
        if (t_max <= t_min) return false;

        // Z
        double invDz = 1.0 / r.direction.z;
        double tz0 = (min().z - r.origin.z) * invDz;
        double tz1 = (max().z - r.origin.z) * invDz;
        if (invDz < 0.0) std::swap(tz0, tz1);
        t_min = tz0 > t_min ? tz0 : t_min;
        t_max = tz1 < t_max ? tz1 : t_max;
        return t_max > t_min;
    }
};

inline AABB surrounding_box(const AABB& a, const AABB& b){
    Vec3 small(std::min(a.min().x, b.min().x),
               std::min(a.min().y, b.min().y),
               std::min(a.min().z, b.min().z));
    Vec3 big  (std::max(a.max().x, b.max().x),
               std::max(a.max().y, b.max().y),
               std::max(a.max().z, b.max().z));
    return AABB(small, big);
}
