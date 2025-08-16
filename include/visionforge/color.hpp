#pragma once
#include "visionforge/vec3.hpp"
#include <cstdint>
#include <array>
#include <algorithm>

// --- Canonical aliases ---
using color  = Vec3;  // linear RGB
using point3 = Vec3;  // 3D point

// --- Helpers ---
inline color lerp(const color& a, const color& b, double t) { return a*(1.0 - t) + b*t; }
inline color hadamard(const color& a, const color& b)       { return a * b; } // component-wise

inline double saturate(double x) { return std::min(1.0, std::max(0.0, x)); }
inline color  saturate(const color& c) { return color(saturate(c.x), saturate(c.y), saturate(c.z)); }

// Linear ↔ sRGB (approx.) — use if you ever need UI colors or PNG export.
// Your path-tracer should work in linear; sRGB is for IO/display only.
inline double srgb_from_linear_1(double x) {
    return (x <= 0.0031308) ? 12.92*x : 1.055*std::pow(x, 1.0/2.4) - 0.055;
}
inline double linear_from_srgb_1(double x) {
    return (x <= 0.04045) ? x/12.92 : std::pow((x + 0.055)/1.055, 2.4);
}
inline color srgb_from_linear(const color& c) {
    return color(srgb_from_linear_1(c.x), srgb_from_linear_1(c.y), srgb_from_linear_1(c.z));
}
inline color linear_from_srgb(const color& c) {
    return color(linear_from_srgb_1(c.x), linear_from_srgb_1(c.y), linear_from_srgb_1(c.z));
}

// Gamma 2.2 shortcuts (if you prefer simple gamma)
inline color gamma_encode(const color& c, double gamma=2.2) {
    return color(std::pow(c.x, 1.0/gamma), std::pow(c.y, 1.0/gamma), std::pow(c.z, 1.0/gamma));
}
inline color gamma_decode(const color& c, double gamma=2.2) {
    return color(std::pow(c.x, gamma), std::pow(c.y, gamma), std::pow(c.z, gamma));
}

// Pack to 8-bit (assumes input is already tonemapped/gamma’d if needed)
inline std::array<uint8_t,3> to_rgb8(const color& c) {
    color s = saturate(c);
    return { uint8_t(255.999*s.x), uint8_t(255.999*s.y), uint8_t(255.999*s.z) };
}
