#pragma once
#include <cstdint>

namespace vf {
class PCG32 {
    uint64_t state_ = 0x853c49e6748fea9bULL;
    uint64_t inc_   = 0xda3e39cb94b95bdbULL;
public:
    PCG32() = default;
    PCG32(uint64_t initstate, uint64_t initseq){ seed(initstate, initseq); }
    void seed(uint64_t initstate, uint64_t initseq){
        state_ = 0U; inc_ = (initseq<<1u) | 1u;
        operator()(); state_ += initstate; operator()();
    }
    uint32_t operator()() {
        uint64_t oldstate = state_;
        state_ = oldstate * 6364136223846793005ULL + (inc_ | 1);
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    double uniform_double(){ return (operator()() >> 11) * (1.0/9007199254740992.0); } // [0,1)
};
} // namespace vf
