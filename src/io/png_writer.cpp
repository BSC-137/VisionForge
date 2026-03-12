#include "visionforge/png_writer.hpp"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include <zlib.h>

namespace {

constexpr uint8_t kPngSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};

void write_u32_be(FILE* f, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>((v >> 24) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>(v & 0xFF)
    };
    std::fwrite(b, 1, 4, f);
}

bool write_chunk(FILE* f, const char type[4], const uint8_t* data, uint32_t size) {
    write_u32_be(f, size);
    std::fwrite(type, 1, 4, f);
    if (size > 0) std::fwrite(data, 1, size, f);

    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(type), 4);
    if (size > 0) crc = crc32(crc, data, size);
    write_u32_be(f, static_cast<uint32_t>(crc));
    return std::ferror(f) == 0;
}

} // namespace

namespace vf {

bool write_png_rgb8(const char* path, int w, int h, const unsigned char* rgb) {
    if (!path || !rgb || w <= 0 || h <= 0) return false;

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;

    if (std::fwrite(kPngSignature, 1, 8, f) != 8) {
        std::fclose(f);
        return false;
    }

    uint8_t ihdr[13] = {};
    ihdr[0] = static_cast<uint8_t>((w >> 24) & 0xFF);
    ihdr[1] = static_cast<uint8_t>((w >> 16) & 0xFF);
    ihdr[2] = static_cast<uint8_t>((w >> 8) & 0xFF);
    ihdr[3] = static_cast<uint8_t>(w & 0xFF);
    ihdr[4] = static_cast<uint8_t>((h >> 24) & 0xFF);
    ihdr[5] = static_cast<uint8_t>((h >> 16) & 0xFF);
    ihdr[6] = static_cast<uint8_t>((h >> 8) & 0xFF);
    ihdr[7] = static_cast<uint8_t>(h & 0xFF);
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    if (!write_chunk(f, "IHDR", ihdr, sizeof(ihdr))) {
        std::fclose(f);
        return false;
    }

    const size_t row_bytes = static_cast<size_t>(w) * 3;
    std::vector<uint8_t> raw(static_cast<size_t>(h) * (row_bytes + 1), 0);
    for (int y = 0; y < h; ++y) {
        uint8_t* dst = raw.data() + static_cast<size_t>(y) * (row_bytes + 1);
        dst[0] = 0;
        const unsigned char* src = rgb + static_cast<size_t>(y) * row_bytes;
        std::memcpy(dst + 1, src, row_bytes);
    }

    uLongf comp_bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> compressed(comp_bound);
    if (compress2(compressed.data(), &comp_bound, raw.data(), static_cast<uLong>(raw.size()), Z_BEST_SPEED) != Z_OK) {
        std::fclose(f);
        return false;
    }
    compressed.resize(comp_bound);

    if (!write_chunk(f, "IDAT", compressed.data(), static_cast<uint32_t>(compressed.size()))) {
        std::fclose(f);
        return false;
    }
    if (!write_chunk(f, "IEND", nullptr, 0)) {
        std::fclose(f);
        return false;
    }

    bool ok = std::fflush(f) == 0 && std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

} // namespace vf
