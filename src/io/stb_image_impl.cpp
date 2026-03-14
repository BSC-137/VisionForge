#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_HDR
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>
#include "visionforge/image_texture.hpp"

float* ImageTexture::load_float_image(const char* path, int* w, int* h, int* n) {
    return stbi_loadf(path, w, h, n, 0);
}
void ImageTexture::free_float_image(float* p) {
    stbi_image_free(p);
}
