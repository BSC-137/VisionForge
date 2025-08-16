#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include "visionforge/color.hpp"  // brings point3 = Vec3
#include "visionforge/vec3.hpp"

namespace vf {

class perlin {
public:
    explicit perlin(uint32_t seed = 1337U) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < POINT_COUNT; ++i) {
            // random unit gradient vectors
            Vec3 v(dist(rng), dist(rng), dist(rng));
            ranvec[i] = normalize(v);
        }
        for (int i = 0; i < POINT_COUNT; ++i) {
            perm_x[i] = perm_y[i] = perm_z[i] = i;
        }
        auto shuffle = [&](std::array<int,POINT_COUNT>& p){
            for (int i = POINT_COUNT - 1; i > 0; --i) {
                std::uniform_int_distribution<int> d(0, i);
                std::swap(p[i], p[d(rng)]);
            }
        };
        shuffle(perm_x); shuffle(perm_y); shuffle(perm_z);
    }

    // Classic Perlin: trilinear interpolation of gradient-dot products
    double noise(const point3& p) const {
        int i = (int)std::floor(p.x);
        int j = (int)std::floor(p.y);
        int k = (int)std::floor(p.z);

        Vec3 c[2][2][2];
        for (int di=0; di<2; ++di)
            for (int dj=0; dj<2; ++dj)
                for (int dk=0; dk<2; ++dk) {
                    int idx = perm_x[(i+di)&255] ^ perm_y[(j+dj)&255] ^ perm_z[(k+dk)&255];
                    c[di][dj][dk] = ranvec[idx];
                }

        double u = p.x - i, v = p.y - j, w = p.z - k;
        auto fade = [](double t){ return t*t*(3 - 2*t); };
        double uu = fade(u), vv = fade(v), ww = fade(w);

        double accum = 0.0;
        for (int di=0; di<2; ++di)
            for (int dj=0; dj<2; ++dj)
                for (int dk=0; dk<2; ++dk) {
                    Vec3 weight(u - di, v - dj, w - dk);
                    double d = dot(c[di][dj][dk], weight);
                    accum += (di?uu:1-uu) * (dj?vv:1-vv) * (dk?ww:1-ww) * d;
                }
        return accum; // ~[-1,1]
    }

    // Fractal Brownian Motion (few octaves for speed)
    double fbm(const point3& p, int octaves=5, double gain=0.5, double lacunarity=2.0) const {
        double amp = 1.0, freq = 1.0, sum = 0.0, norm = 0.0;
        for (int o=0; o<octaves; ++o) {
            sum  += amp * noise(p * freq);
            norm += amp;
            amp  *= gain;
            freq *= lacunarity;
        }
        return sum / (norm > 0.0 ? norm : 1.0);
    }

    // Numerical gradient for bump mapping dunes
    Vec3 grad(const point3& p, double eps=1e-3) const {
        double nx = fbm(point3(p.x + eps, p.y, p.z)) - fbm(point3(p.x - eps, p.y, p.z));
        double ny = fbm(point3(p.x, p.y + eps, p.z)) - fbm(point3(p.x, p.y - eps, p.z));
        double nz = fbm(point3(p.x, p.y, p.z + eps)) - fbm(point3(p.x, p.y, p.z - eps));
        return Vec3(nx, ny, nz) / (2.0 * eps);
    }

private:
    static constexpr int POINT_COUNT = 256;
    std::array<Vec3, POINT_COUNT> ranvec;
    std::array<int,  POINT_COUNT> perm_x, perm_y, perm_z;
};

} // namespace vf
