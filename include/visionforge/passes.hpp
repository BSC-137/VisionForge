#pragma once
#include <vector>
#include <cstdint>

struct GBuffer {
    int width=0, height=0;
    std::vector<uint32_t> inst_id;   // per-pixel instance ids

    GBuffer() = default;
    GBuffer(int w,int h) : width(w),height(h),inst_id(w*h,0) {}
    inline uint32_t& at(int x,int y) { return inst_id[y*width + x]; }
    inline const uint32_t& at(int x,int y) const { return inst_id[y*width + x]; }
};
