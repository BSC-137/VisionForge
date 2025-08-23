#pragma once
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "visionforge/passes.hpp"

// Simple 8-bit PGM; ids >255 are clamped for now.
inline void write_inst_pgm(const std::string& path, const GBuffer& g) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f << "P5\n" << g.width << " " << g.height << "\n255\n";
    std::vector<unsigned char> row(g.width);
    for (int y=0; y<g.height; ++y) {
        for (int x=0; x<g.width; ++x) {
            uint32_t id = g.inst_id[y*g.width + x];
            row[x] = static_cast<unsigned char>(id > 255 ? 255 : id);
        }
        f.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
}
