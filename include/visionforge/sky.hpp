#pragma once
#include <cmath>
#include <algorithm>
#include "visionforge/vec3.hpp"

// Fallback if PI isn't provided elsewhere.
#ifndef PI
#define PI 3.14159265358979323846
#endif

// Simple analytic morning sky: blue gradient + forward-scatter + sun disc + warm halo.
// azimuth: degrees clockwise around Y from +X (right-handed)
// elevation: degrees above horizon
class Sky {
public:
    // strength boosts the whole dome so it remains visible after exposure/ACES
    explicit Sky(double sun_az_deg = 300.0,
                 double sun_elev_deg = 12.0,
                 double turbidity = 3.5,
                 double strength = 4.0)
    : turb_(turbidity), gain_(strength)
    {
        set_sun_dir(sun_az_deg, sun_elev_deg);
    }

    // Change sun direction later if needed
    void set_sun_dir(double sun_az_deg, double sun_elev_deg) {
        const double az = sun_az_deg  * PI/180.0;
        const double el = sun_elev_deg * PI/180.0;
        // Y-up, azimuth around Y
        sun_dir_ = normalize(Vec3(std::cos(el)*std::cos(az),
                                  std::sin(el),
                                  std::cos(el)*std::sin(az)));
    }

    // Evaluate radiance toward 'dir' (world space). 'dir' need not be normalized.
    Vec3 eval(const Vec3& dir_ws) const {
        const Vec3 dir = normalize(dir_ws);

        // Horizon↔zenith gradient
        const double t = 0.5 * (dir.y + 1.0);
        const Vec3 horizon(0.80, 0.86, 0.98);
        const Vec3 zenith (0.22, 0.45, 0.95);
        const Vec3 skycol = (1.0 - t) * horizon + t * zenith;

        // Blue forward-scattering lobe (very lightweight Rayleigh-ish)
        const double mu = std::max(0.0, dot(dir, sun_dir_)); // cos(theta to sun)
        const double rayleigh = std::pow(mu, 4.0);

        // Sun disc (~0.23° sigma) and warm halo (~3.4° sigma)
        const double mu_clamped = std::clamp(mu, 0.0, 1.0);
        const double sun_angle = std::acos(mu_clamped);

        const double disc_sigma = 0.004; // radians
        const double glow_sigma = 0.06;  // radians

        const double disc = std::exp(-(sun_angle*sun_angle) / (2.0 * disc_sigma * disc_sigma));
        const double glow = std::exp(-(sun_angle*sun_angle) / (2.0 * glow_sigma * glow_sigma));

        const Vec3 warm(1.0, 0.75, 0.50);

        // Combine; weights tuned for your exposure + ACES + sqrt gamma
        Vec3 result = 1.3 * skycol
                    + 0.6 * rayleigh * Vec3(0.5, 0.6, 1.0)
                    + (12000.0 * disc + 30.0 * glow) * warm;

        return gain_ * result; // global boost so the sky shows up in your pipeline
    }

private:
    Vec3  sun_dir_;
    double turb_;   // reserved for future use
    double gain_;   // overall intensity scale
};
