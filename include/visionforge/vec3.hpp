#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

// ---------- constants ----------
constexpr double PI = 3.14159265358979323846;

// ---------- RNG ----------
inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}
inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

// ---------- Vec3 ----------
class Vec3 {
public:
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator-() const { return Vec3(-x, -y, -z); }

    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator*=(double t)      { x*=t;   y*=t;   z*=t;   return *this; }
    Vec3& operator/=(double t)      { return *this *= 1.0/t; }

    double length() const           { return std::sqrt(length_squared()); }
    double length_squared() const   { return x*x + y*y + z*z; }
};

inline Vec3 operator+(const Vec3& a, const Vec3& b){ return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b){ return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline Vec3 operator*(const Vec3& a, const Vec3& b){ return Vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline Vec3 operator*(double t, const Vec3& v)     { return Vec3(t*v.x, t*v.y, t*v.z); }
inline Vec3 operator*(const Vec3& v, double t)     { return t * v; }
inline Vec3 operator/(Vec3 v, double t)            { return (1.0/t) * v; }

inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3   cross(const Vec3& a, const Vec3& b){
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline Vec3   normalize(Vec3 v){ return v / v.length(); }

// ---------- sampling helpers ----------
inline Vec3 random_in_unit_sphere() {
    while (true) {
        Vec3 p(random_double(-1,1), random_double(-1,1), random_double(-1,1));
        if (p.length_squared() < 1.0) return p;
    }
}
inline Vec3 random_unit_vector() { return normalize(random_in_unit_sphere()); }

inline Vec3 random_in_unit_disk() {        // for depth-of-field lens sampling
    while (true) {
        Vec3 p(random_double(-1,1), random_double(-1,1), 0.0);
        if (p.length_squared() < 1.0) return p;
    }
}

inline bool near_zero(const Vec3& v){
    const double s = 1e-8;
    return (std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s);
}

// ---------- optics ----------
inline Vec3 reflect(const Vec3& v, const Vec3& n){
    return v - 2.0 * dot(v, n) * n;
}
inline Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat){
    double cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp  = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_par   = -std::sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_par;
}

#endif
