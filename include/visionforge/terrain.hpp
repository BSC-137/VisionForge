#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include "visionforge/vec3.hpp"
#include "visionforge/perlin.hpp"
#include "visionforge/hittable_list.hpp"
#include "visionforge/triangle.hpp"
#include "visionforge/material.hpp"

struct TerrainSample {
    double height;
    Vec3   normal;
};

struct HeightField {
    double xmin, xmax, zmin, zmax;
    int    nx, nz;
    double amp, scale;
    vf::perlin noise;
    std::vector<double> h;

    HeightField(double X0, double X1, double Z0, double Z1,
                int NX, int NZ, double A, double S, uint32_t seed = 1337)
        : xmin(X0), xmax(X1), zmin(Z0), zmax(Z1),
          nx(NX), nz(NZ), amp(A), scale(S), noise(seed),
          h(static_cast<size_t>(NX + 1) * static_cast<size_t>(NZ + 1)) {}

    inline int idx(int i, int k) const { return k * (nx + 1) + i; }

    void generate() {
        auto ridged_fbm = [&](const Vec3& p) {
            double a = 1.0, f = 1.0, s = 0.0, nrm = 0.0;
            for (int o = 0; o < 5; ++o) {
                double n = 1.0 - std::fabs(noise.fbm(p * f, 1, 0.5, 2.0));
                n = 0.85 * n + 0.15 * n * n;
                s += a * n; nrm += a; a *= 0.62; f *= 1.9;
            }
            return s / std::max(1e-8, nrm);
        };
        auto warp = [&](const Vec3& p) {
            return p + 0.35 * Vec3(
                noise.fbm(p * 1.7, 3, 0.5, 2.1),
                noise.fbm(p * 2.0, 3, 0.5, 2.0),
                noise.fbm(p * 1.8, 3, 0.5, 2.2));
        };
        for (int k = 0; k <= nz; ++k) {
            double z = zmin + (zmax - zmin) * (double(k) / nz);
            for (int i = 0; i <= nx; ++i) {
                double x = xmin + (xmax - xmin) * (double(i) / nx);
                Vec3 pd = warp(Vec3(x * scale, 0.0, z * scale));
                double dunes = ridged_fbm(pd);
                double mid   = 0.25 * noise.fbm(pd * 2.4, 4, 0.5, 2.0);
                double fine  = 0.06 * noise.fbm(pd * 10.0, 3, 0.5, 2.0);
                h[idx(i, k)] = amp * ((1.1 * dunes - 0.55) + mid + fine);
            }
        }
    }

    double height_at(double x, double z) const {
        double u = (x - xmin) / (xmax - xmin), v = (z - zmin) / (zmax - zmin);
        u = std::clamp(u, 0.0, 1.0);
        v = std::clamp(v, 0.0, 1.0);
        double fx = u * nx, fz = v * nz;
        int i = std::clamp(static_cast<int>(std::floor(fx)), 0, nx - 1);
        int k = std::clamp(static_cast<int>(std::floor(fz)), 0, nz - 1);
        double du = fx - i, dv = fz - k;
        auto H = [&](int ii, int kk) { return h[idx(ii, kk)]; };
        double h00 = H(i, k),     h10 = H(i + 1, k);
        double h01 = H(i, k + 1), h11 = H(i + 1, k + 1);
        double h0  = h00 * (1 - du) + h10 * du;
        double h1  = h01 * (1 - du) + h11 * du;
        return h0 * (1 - dv) + h1 * dv;
    }

    Vec3 normal_at(int i, int k) const {
        auto H = [&](int ii, int kk) {
            ii = std::clamp(ii, 0, nx);
            kk = std::clamp(kk, 0, nz);
            return h[idx(ii, kk)];
        };
        double hx = (H(i + 1, k) - H(i - 1, k)) * 0.5;
        double hz = (H(i, k + 1) - H(i, k - 1)) * 0.5;
        double dx = (xmax - xmin) / nx, dz = (zmax - zmin) / nz;
        Vec3 tx(dx, hx, 0), tz(0, hz, dz);
        return normalize(cross(tz, tx));
    }

    inline Vec3 normal_at_world(double x, double z) const {
        const double eps_x = (xmax - xmin) / nx;
        const double eps_z = (zmax - zmin) / nz;
        const double hL = height_at(x - eps_x, z);
        const double hR = height_at(x + eps_x, z);
        const double hD = height_at(x, z - eps_z);
        const double hU = height_at(x, z + eps_z);
        Vec3 tx(2.0 * eps_x, hR - hL, 0.0);
        Vec3 tz(0.0, hU - hD, 2.0 * eps_z);
        return normalize(cross(tz, tx));
    }

    inline TerrainSample get_terrain_height_and_normal(double x, double z) const {
        return { height_at(x, z), normal_at_world(x, z) };
    }

    void carve_footprint(double x0, double z0, double edge, double sink, double rim_amp) {
        double r = 0.6 * edge;
        double r2 = r * r;
        double r_out = r * 1.8;
        for (int k = 0; k <= nz; ++k) {
            double z = zmin + (zmax - zmin) * (double(k) / nz);
            for (int i = 0; i <= nx; ++i) {
                double x = xmin + (xmax - xmin) * (double(i) / nx);
                double dx = x - x0, dz = z - z0;
                double d2 = dx * dx + dz * dz;
                if (d2 > r_out * r_out) continue;
                double depress = sink * std::exp(-(d2) / (2.0 * r2));
                double ring = std::exp(-std::pow((std::sqrt(d2) / r - 1.1) / 0.18, 2.0));
                double berm = rim_amp * ring;
                h[idx(i, k)] -= depress;
                h[idx(i, k)] += berm;
            }
        }
    }

    void to_triangles(HittableList& out, const std::shared_ptr<Material>& m) const {
        for (int k = 0; k < nz; ++k) {
            double z0 = zmin + (zmax - zmin) * (double(k) / nz);
            double z1 = z0 + (zmax - zmin) / nz;
            for (int i = 0; i < nx; ++i) {
                double x0 = xmin + (xmax - xmin) * (double(i) / nx);
                double x1 = x0 + (xmax - xmin) / nx;
                Vec3 v00(x0, h[idx(i, k)],       z0), v10(x1, h[idx(i + 1, k)],     z0);
                Vec3 v01(x0, h[idx(i, k + 1)],   z1), v11(x1, h[idx(i + 1, k + 1)], z1);
                Vec3 n00 = normal_at(i, k),       n10 = normal_at(i + 1, k),
                     n01 = normal_at(i, k + 1),   n11 = normal_at(i + 1, k + 1);
                if (((i ^ k) & 1) == 0) {
                    out.add(std::make_shared<Triangle>(v00, v10, v11, n00, n10, n11, m));
                    out.add(std::make_shared<Triangle>(v00, v11, v01, n00, n11, n01, m));
                } else {
                    out.add(std::make_shared<Triangle>(v00, v10, v01, n00, n10, n01, m));
                    out.add(std::make_shared<Triangle>(v10, v11, v01, n10, n11, n01, m));
                }
            }
        }
    }
};
