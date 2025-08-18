#pragma once
#include <cmath>
#include <algorithm>
#include "visionforge/vec3.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Bright summer sky (camera background only) with a soft visible sun.
// Does NOT contribute to scene lighting; your rect area-light does that.
// We align the visible sun with the rect light's normal in main.cpp.
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

    // set by azimuth/elevation (degrees)
    void set_sun_dir(double sun_az_deg, double sun_elev_deg) {
        const double az = sun_az_deg  * PI/180.0;
        const double el = sun_elev_deg * PI/180.0;
        sun_dir_ = normalize(Vec3(std::cos(el)*std::cos(az),
                                  std::sin(el),
                                  std::cos(el)*std::sin(az)));
    }

    // set directly from a world-space direction (e.g., rect-light normal)
    void set_sun_from_dir(const Vec3& d) { sun_dir_ = normalize(d); }

    // radiance toward 'dir'
    Vec3 eval(const Vec3& dir_ws) const {
        const Vec3 dir = normalize(dir_ws);

        // Horizon↔zenith gradient (bright, clean blue)
        const double t = 0.5 * (dir.y + 1.0);            // 0 at nadir, 1 at zenith
        const Vec3 horizon(0.86, 0.93, 1.00);            // pale blue near horizon
        const Vec3 zenith (0.38, 0.68, 1.05);            // deeper blue upward
        const Vec3 skycol = (1.0 - t) * horizon + t * zenith;

        // Gentle forward scattering to bias toward sun
        const double mu = std::max(0.0, dot(dir, sun_dir_));
        const double rayleigh = std::pow(mu, 3.0);
        const Vec3 rayleigh_tint(0.55, 0.70, 1.0);

        // Soft sun disc & warm halo
        const double mu_c = std::clamp(mu, 0.0, 1.0);
        const double ang  = std::acos(mu_c);
        const double disc_sigma = 0.004;   // ~0.23°
        const double glow_sigma = 0.080;   // ~4.6°
        const double disc = std::exp(-(ang*ang)/(2.0*disc_sigma*disc_sigma));
        const double glow = std::exp(-(ang*ang)/(2.0*glow_sigma*glow_sigma));
        const Vec3 warm(1.0, 0.82, 0.55);
        const double disc_amp = 90.0;      // visible but not blown out after ACES
        const double glow_amp = 8.0;

        Vec3 result = 1.4 * skycol
                    + 0.30 * rayleigh * rayleigh_tint
                    + (disc_amp * disc + glow_amp * glow) * warm;

        return gain_ * result;
    }

private:
    Vec3  sun_dir_;
    double turb_;
    double gain_;
};
