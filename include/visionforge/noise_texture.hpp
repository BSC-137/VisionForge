#pragma once
#include "visionforge/texture.hpp"
#include "visionforge/perlin.hpp"  // your Perlin class

namespace vf {

class sand_texture : public texture {
public:
    sand_texture(double scale=3.5,
                 color c1={0.76,0.69,0.55},
                 color c2={0.66,0.60,0.47},
                 uint32_t seed=1337U)
        : m_scale(scale), col1(c1), col2(c2), m_perlin(seed) {}

    color value(double, double, const point3& p) const override {
        double n = 0.5 * (1.0 + m_perlin.fbm(p * m_scale, /*oct=*/5, 0.5, 2.1)); // [0,1]
        double speck = 0.03 * m_perlin.noise(p*25.0);
        return lerp(col1, col2, n) + color(speck, speck, speck);
    }

    const perlin& perlin_ref() const { return m_perlin; }
    double scale() const { return m_scale; }

private:
    double m_scale;
    color  col1, col2;
    perlin m_perlin;
};

} // namespace vf
