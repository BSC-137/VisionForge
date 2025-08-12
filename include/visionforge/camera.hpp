#pragma once
#include "ray.hpp"

class Camera {
public:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;           // camera basis
    double lens_radius;     // aperture/2
    double time0, time1;    // shutter open/close

    Camera(
        Vec3 lookfrom   = Vec3(3,3,2),
        Vec3 lookat     = Vec3(0,0,-1),
        Vec3 vup        = Vec3(0,1,0),
        double vfov_deg = 45.0,       // vertical FOV in degrees
        double aspect   = 2.0,        // width/height
        double aperture = 0.0,        // 0 => pinhole
        double focus_dist = 3.4,      // distance to focus plane
        double t0 = 0.0,              // shutter open
        double t1 = 1.0               // shutter close
    ) : lens_radius(aperture * 0.5), time0(t0), time1(t1)
    {
        double theta = vfov_deg * (PI / 180.0);
        double h = tan(theta / 2.0);
        double viewport_height = 2.0 * h;
        double viewport_width  = aspect * viewport_height;

        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width  * u;
        vertical   = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal*0.5 - vertical*0.5 - focus_dist * w;
    }

    Ray get_ray(double s, double t) const {
        // Depth of field: sample a disk aperture
        Vec3 rd = lens_radius * random_in_unit_disk();
        Vec3 offset = u * rd.x + v * rd.y;

        // Motion blur: sample time in shutter interval
        double time = random_double(time0, time1);

        return Ray(
            origin + offset,
            lower_left_corner + s*horizontal + t*vertical - origin - offset,
            time
        );
    }
};
