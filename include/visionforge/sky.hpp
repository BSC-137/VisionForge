#pragma once
#include <cmath>
#include <algorithm>
#include "visionforge/vec3.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Camera background sky with a visible sun disc + warm halo.
// NOTE: Background only (no lighting). Your rect area light still illuminates the scene.
// main.cpp can align the visible sun with the rect light's normal.
class Sky {
public:
    explicit Sky(double sun_az_deg = 300.0,
                 double sun_elev_deg = 12.0,
                 double turbidity = 3.5,
                 double strength = 1.0)
    : turb_(turbidity), gain_(strength) {
        set_sun_dir(sun_az_deg, sun_elev_deg);
    }

    // Overall intensity
    void set_gain(double g) { gain_ = g; }

    // Convenience alias for CLI: set by azimuth/elevation (degrees)
    void set_angles(double sun_az_deg, double sun_elev_deg) { set_sun_dir(sun_az_deg, sun_elev_deg); }

    // Allow CLI to change turbidity (subtle/unused unless you fold it into eval)
    void set_turbidity(double t) { turb_ = t; }

    // Set by azimuth/elevation (degrees)
    void set_sun_dir(double sun_az_deg, double sun_elev_deg) {
        const double az = sun_az_deg  * PI/180.0;
        const double el = sun_elev_deg * PI/180.0;
        sun_dir_ = normalize(Vec3(std::cos(el)*std::cos(az),
                                  std::sin(el),
                                  std::cos(el)*std::sin(az)));
    }

    // Set directly from a world-space direction (e.g., rect-light normal)
    void set_sun_from_dir(const Vec3& d) { sun_dir_ = normalize(d); }

    // Background radiance toward dir_ws
    Vec3 eval(const Vec3& dir_ws) const {
        const Vec3 dir = normalize(dir_ws);

        // Blue gradient: t = 0 at nadir, 1 at zenith
        const double t = 0.5 * (dir.y + 1.0);
        const Vec3 horizon(0.72, 0.82, 1.00);
        const Vec3 zenith (0.22, 0.48, 1.15);
        Vec3 skycol = (1.0 - t) * horizon + t * zenith;

        // Subtle forward Rayleigh scatter toward the sun
        const double mu = std::max(0.0, dot(dir, sun_dir_));
        const double rayleigh = std::pow(mu, 2.5);
        const Vec3 rayleigh_tint(0.55, 0.70, 1.0);
        skycol += 0.25 * rayleigh * rayleigh_tint;

        // Sun disc + warm halo
        const double mu_c = std::clamp(mu, 0.0, 1.0);
        const double ang  = std::acos(mu_c);

        // Disc ~0.5° FWHM, halo ~5°
        const double disc_sigma = 0.0045;
        const double glow_sigma = 0.085;

        const double disc = std::exp(-(ang*ang) / (2.0 * disc_sigma * disc_sigma));
        const double glow = std::exp(-(ang*ang) / (2.0 * glow_sigma * glow_sigma));

        const Vec3 warm(1.00, 0.84, 0.62);

        // Tuned to look good after ACES tonemap + sqrt "gamma"
        const double disc_amp = 110.0;
        const double glow_amp = 9.0;

        Vec3 result = skycol + (disc_amp * disc + glow_amp * glow) * warm;
        return gain_ * result;
    }

private:
    Vec3  sun_dir_{0,1,0};
    double turb_{3.5};
    double gain_{1.0};
};
