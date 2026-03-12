#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>

struct GBuffer {
    int width=0, height=0;
    std::vector<uint32_t> inst_id;   // per-pixel instance ids
    std::vector<float> depth;        // linear distance from camera (metres)
    std::vector<float> normal_x;     // world-space hit normal X
    std::vector<float> normal_y;     // world-space hit normal Y
    std::vector<float> normal_z;     // world-space hit normal Z

    GBuffer() = default;
    GBuffer(int w, int h)
        : width(w), height(h),
          inst_id(w*h, 0),
          depth(w*h, 0.0f),
          normal_x(w*h, 0.0f),
          normal_y(w*h, 0.0f),
          normal_z(w*h, 0.0f) {}

    void clear() {
        const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::fill(inst_id.begin(), inst_id.begin() + n, 0u);
        std::fill(depth.begin(), depth.begin() + n, 0.0f);
        std::fill(normal_x.begin(), normal_x.begin() + n, 0.0f);
        std::fill(normal_y.begin(), normal_y.begin() + n, 0.0f);
        std::fill(normal_z.begin(), normal_z.begin() + n, 0.0f);
    }

    inline uint32_t& at(int x,int y) { return inst_id[y*width + x]; }
    inline const uint32_t& at(int x,int y) const { return inst_id[y*width + x]; }
};
