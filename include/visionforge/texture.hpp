#pragma once
#include "visionforge/color.hpp"

namespace vf {

class texture {
public:
    virtual ~texture() = default;
    virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
public:
    solid_color() : color_value(0,0,0) {}
    explicit solid_color(const color& c) : color_value(c) {}
    solid_color(double r, double g, double b) : color_value(r,g,b) {}
    color value(double, double, const point3&) const override { return color_value; }
private:
    color color_value;
};

} // namespace vf
