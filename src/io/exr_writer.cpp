// Force TinyEXR to use system zlib (not miniz)
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_ZLIB  1

#define TINYEXR_IMPLEMENTATION
#include <zlib.h>        // ensure zlib types like uLong, Bytef are visible
#include <tinyexr.h>

#include <cstring>
#include "visionforge/exr_writer.hpp"
#include "visionforge/passes.hpp"


namespace vf {

static void split_rgb(const std::vector<Vec3>& rgb, int w, int h,
                      std::vector<float>& R, std::vector<float>& G, std::vector<float>& B){
    R.resize(w*h); G.resize(w*h); B.resize(w*h);
    for (int i=0;i<w*h;++i){ R[i]=(float)rgb[i].x; G[i]=(float)rgb[i].y; B[i]=(float)rgb[i].z; }
}

bool write_rgb_exr(const char* path, int w, int h, const std::vector<Vec3>& rgb){
    std::vector<float> R,G,B; split_rgb(rgb,w,h,R,G,B);

    EXRHeader header; InitEXRHeader(&header);
    EXRImage image;  InitEXRImage(&image);
    image.num_channels = 3;

    std::vector<float*> images(3);
    images[0] = B.data(); images[1] = G.data(); images[2] = R.data(); // BGR
    image.images = reinterpret_cast<unsigned char**>(images.data());
    image.width  = w; image.height = h;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo)*3);
    std::strcpy(header.channels[0].name, "B");
    std::strcpy(header.channels[1].name, "G");
    std::strcpy(header.channels[2].name, "R");

    header.pixel_types = (int*)malloc(sizeof(int)*3);
    header.requested_pixel_types = (int*)malloc(sizeof(int)*3);
    for (int i=0;i<3;++i){ header.pixel_types[i]=TINYEXR_PIXELTYPE_FLOAT; header.requested_pixel_types[i]=TINYEXR_PIXELTYPE_FLOAT; }

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);

    if (ret != TINYEXR_SUCCESS){
        if (err){ FreeEXRErrorMessage(err); }
        free(header.channels); free(header.pixel_types); free(header.requested_pixel_types);
        return false;
    }
    free(header.channels); free(header.pixel_types); free(header.requested_pixel_types);
    return true;
}

bool write_float_exr(const char* path, int w, int h, const std::vector<float>& img, const char* channel){
    EXRHeader header; InitEXRHeader(&header);
    EXRImage image;  InitEXRImage(&image);
    image.num_channels = 1;

    float* ptr = const_cast<float*>(img.data());
    image.images = reinterpret_cast<unsigned char**>(&ptr);
    image.width = w; image.height = h;

    header.num_channels = 1;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo));
    std::strncpy(header.channels[0].name, channel, 255); header.channels[0].name[255]='\0';

    header.pixel_types = (int*)malloc(sizeof(int));
    header.requested_pixel_types = (int*)malloc(sizeof(int));
    header.pixel_types[0]=TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[0]=TINYEXR_PIXELTYPE_FLOAT;

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);
    if (ret != TINYEXR_SUCCESS){
        if (err){ FreeEXRErrorMessage(err); }
        free(header.channels); free(header.pixel_types); free(header.requested_pixel_types);
        return false;
    }
    free(header.channels); free(header.pixel_types); free(header.requested_pixel_types);
    return true;
}

bool write_gbuffer_exr(const char* path, const GBuffer& gbuf) {
    const int w = gbuf.width;
    const int h = gbuf.height;
    constexpr int NC = 4;

    EXRHeader header; InitEXRHeader(&header);
    EXRImage image;   InitEXRImage(&image);
    image.num_channels = NC;
    image.width  = w;
    image.height = h;

    // TinyEXR requires channels sorted alphabetically
    // Depth < Normal.X < Normal.Y < Normal.Z
    float* ch_ptrs[NC] = {
        const_cast<float*>(gbuf.depth.data()),
        const_cast<float*>(gbuf.normal_x.data()),
        const_cast<float*>(gbuf.normal_y.data()),
        const_cast<float*>(gbuf.normal_z.data()),
    };
    image.images = reinterpret_cast<unsigned char**>(ch_ptrs);

    header.num_channels = NC;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * NC);
    std::strcpy(header.channels[0].name, "Depth");
    std::strcpy(header.channels[1].name, "Normal.X");
    std::strcpy(header.channels[2].name, "Normal.Y");
    std::strcpy(header.channels[3].name, "Normal.Z");

    header.pixel_types           = (int*)malloc(sizeof(int) * NC);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * NC);
    for (int i = 0; i < NC; ++i) {
        header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);
        return false;
    }
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
    return true;
}

} // namespace vf
