#ifndef RAY_H
#define RAY_H

#include "vec3.hpp"

class Ray {
public:
    Vec3 origin;
    Vec3 direction;
    double time; // NEW: time the ray was generated (for motion blur)

    Ray() : time(0.0) {}
    Ray(const Vec3& origin, const Vec3& direction, double time = 0.0)
        : origin(origin), direction(direction), time(time) {}

    Vec3 at(double t) const { return origin + t * direction; }
};

#endif
