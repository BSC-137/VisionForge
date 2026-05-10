#pragma once
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "visionforge/vec3.hpp"

struct stbi_deleter { void operator()(float* p) const; };

class ImageTexture {
public:
    ImageTexture() = default;

    bool load(const std::string& path) {
        int n = 0;
        float* raw = load_float_image(path.c_str(), &w_, &h_, &n);
        if (!raw) {
            std::cerr << "ImageTexture: failed to load " << path << "\n";
            return false;
        }
        pixels_ = std::make_shared<std::vector<float>>(static_cast<size_t>(w_) * h_ * 3);
        if (n == 3) {
            std::copy(raw, raw + static_cast<size_t>(w_) * h_ * 3, pixels_->data());
        } else if (n == 4) {
            for (int i = 0; i < w_ * h_; ++i) {
                (*pixels_)[i * 3 + 0] = raw[i * 4 + 0];
                (*pixels_)[i * 3 + 1] = raw[i * 4 + 1];
                (*pixels_)[i * 3 + 2] = raw[i * 4 + 2];
            }
        } else if (n == 1) {
            for (int i = 0; i < w_ * h_; ++i) {
                (*pixels_)[i * 3 + 0] = raw[i];
                (*pixels_)[i * 3 + 1] = raw[i];
                (*pixels_)[i * 3 + 2] = raw[i];
            }
        }
        free_float_image(raw);
        std::cerr << "ImageTexture: loaded " << path
                  << " (" << w_ << "x" << h_ << ", " << n << " ch)\n";
        return true;
    }

    bool valid() const { return pixels_ && !pixels_->empty(); }
    int width()  const { return w_; }
    int height() const { return h_; }

    inline Vec3 sample(double u, double v) const {
        u = u - std::floor(u);
        v = v - std::floor(v);
        double fx = u * (w_ - 1);
        double fy = v * (h_ - 1);
        int ix = static_cast<int>(fx);
        int iy = static_cast<int>(fy);
        double dx = fx - ix;
        double dy = fy - iy;
        int ix1 = (ix + 1 < w_) ? ix + 1 : 0;
        int iy1 = (iy + 1 < h_) ? iy + 1 : 0;
        Vec3 c00 = fetch(ix,  iy);
        Vec3 c10 = fetch(ix1, iy);
        Vec3 c01 = fetch(ix,  iy1);
        Vec3 c11 = fetch(ix1, iy1);
        Vec3 top = c00 * (1.0 - dx) + c10 * dx;
        Vec3 bot = c01 * (1.0 - dx) + c11 * dx;
        return top * (1.0 - dy) + bot * dy;
    }

private:
    std::shared_ptr<std::vector<float>> pixels_;
    int w_ = 0, h_ = 0;

    inline Vec3 fetch(int x, int y) const {
        const size_t idx = (static_cast<size_t>(y) * w_ + x) * 3;
        const float* p = pixels_->data() + idx;
        return Vec3(p[0], p[1], p[2]);
    }

    static float* load_float_image(const char* path, int* w, int* h, int* n);
    static void   free_float_image(float* p);
};
