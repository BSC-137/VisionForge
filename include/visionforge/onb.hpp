#pragma once
#include "vec3.hpp"

// Build a local frame around a normal so we can sample cosine directions.
struct ONB {
    Vec3 u, v, w; // w = normal

    void build_from_w(const Vec3& n) {
        w = normalize(n);
        Vec3 a = (std::fabs(w.x) > 0.9) ? Vec3(0,1,0) : Vec3(1,0,0);
        v = normalize(cross(w, a));
        u = cross(w, v);
    }
    Vec3 local(const Vec3& a) const { return u*a.x + v*a.y + w*a.z; }
};

// Cosine-weighted hemisphere sample (pdf = cos(theta)/pi)
inline Vec3 random_cosine_direction() {
    double r1 = random_double();
    double r2 = random_double();
    double z  = std::sqrt(1 - r2);
    double phi = 2.0 * PI * r1;
    double x = std::cos(phi) * std::sqrt(r2);
    double y = std::sin(phi) * std::sqrt(r2);
    return Vec3(x, y, z);
}
