#pragma once
#include <cmath>
#include <algorithm>
#include "visionforge/vec3.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

class Sky {
public:
    explicit Sky(double sun_az_deg = 300.0,
                 double sun_elev_deg = 12.0,
                 double turbidity = 3.5,
                 double strength = 1.0)
    : turb_(turbidity), gain_(strength)
    {
        set_sun_dir(sun_az_deg, sun_elev_deg);
    }

    void set_sun_dir(double sun_az_deg, double sun_elev_deg) {
        const double az = sun_az_deg  * PI/180.0;
        const double el = sun_elev_deg * PI/180.0;
        sun_dir_ = normalize(Vec3(std::cos(el)*std::cos(az),
                                  std::sin(el),
                                  std::cos(el)*std::sin(az)));
    }

    Vec3 eval(const Vec3& dir_ws) const {
        const Vec3 dir = normalize(dir_ws);

        // Horizon↔zenith gradient (bright, cyan-tinted)
        const double t = 0.5 * (dir.y + 1.0);
        const Vec3 horizon(0.75, 0.90, 1.00);  // very light blue near horizon
        const Vec3 zenith (0.35, 0.65, 1.00);  // bright azure overhead
        const Vec3 skycol = (1.0 - t) * horizon + t * zenith;

        // Gentle Rayleigh scattering — keeps it “airy”
        const double mu = std::max(0.0, dot(dir, sun_dir_));
        const double rayleigh = std::pow(mu, 3.0);
        const Vec3 rayleigh_tint(0.55, 0.70, 1.0);

        Vec3 result = 1.3 * skycol + 0.35 * rayleigh * rayleigh_tint;

        return gain_ * result;
    }

private:
    Vec3  sun_dir_;
    double turb_;
    double gain_;
};
