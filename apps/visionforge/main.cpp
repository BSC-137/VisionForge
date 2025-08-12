#include <fstream>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <string>
#include "CLI/CLI.hpp"
#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/hittable_list.hpp"
#include "visionforge/sphere.hpp"
#include "visionforge/xy_rect.hpp"
#include "visionforge/xz_rect.hpp"
#include "visionforge/yz_rect.hpp"
#include "visionforge/camera.hpp"
#include "visionforge/material.hpp"
#include "visionforge/lambertian.hpp"
#include "visionforge/metal.hpp"
#include "visionforge/dielectric.hpp"
#include "visionforge/diffuse_light.hpp"
#include "visionforge/onb.hpp"
#include "visionforge/bvh.hpp"
#include "visionforge/pcg32.hpp"
#include "visionforge/exr_writer.hpp"

using namespace std;

static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }
static inline Vec3 aces_tonemap(const Vec3& c){
    const double a=2.51,b=0.03,c2=2.43,d=0.59,e=0.14;
    auto f=[&](double x){ double num=x*(a*x+b), den=x*(c2*x+d)+e; return clamp01(num/den); };
    return {f(c.x),f(c.y),f(c.z)};
}

struct AOV {
    vector<Vec3> beauty;   // linear
    vector<Vec3> albedo;   // first hit
    vector<Vec3> normal;   // world
    vector<float> depth;   // metric (ray t)
    vector<int>   inst_id; // object id (0 = none)
};

struct FirstHit {
    Vec3 albedo{0,0,0};
    Vec3 normal{0,0,0};
    float depth = 0.f;
    int id = 0;
    bool filled = false;
};

static inline bool get_lambert_albedo(const shared_ptr<Material>& m, Vec3& out){
    if(auto* lam = dynamic_cast<Lambertian*>(m.get())){ out = lam->albedo; return true; }
    return false;
}

Vec3 ray_color(const Ray& r, const Hittable& world, int depth, FirstHit* fh){
    if (depth<=0) return Vec3(0,0,0);
    HitRecord rec;
    if (!world.hit(r, 0.001, numeric_limits<double>::infinity(), rec))
        return Vec3(0,0,0);

    if (fh && !fh->filled){
        Vec3 alb; if (get_lambert_albedo(rec.mat, alb)) fh->albedo = alb; else fh->albedo = Vec3(0,0,0);
        fh->normal = rec.normal;
        fh->depth  = (float)rec.t;
        fh->id     = rec.object_id;
        fh->filled = true;
    }

    Vec3 emitted = rec.mat->emitted(rec);
    Ray scattered; Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered))
        return emitted;

    return emitted + attenuation * ray_color(scattered, world, depth-1, nullptr);
}

int main(int argc, char** argv){
    // ---- CLI
    int W=640, H=360, spp=32, max_depth=16;
    uint64_t seed=7;
    string outdir="out";
    CLI::App app{"VisionForge - tiny synthetic data path tracer"};
    app.add_option("--width,-W",  W, "Image width");
    app.add_option("--height,-H", H, "Image height");
    app.add_option("--spp",       spp, "Samples per pixel");
    app.add_option("--max-depth", max_depth, "Path max depth");
    app.add_option("--seed",      seed, "RNG seed");
    app.add_option("-o,--out",    outdir, "Output directory");
    CLI11_PARSE(app, argc, argv);

    // ---- RNG
    vf::PCG32 rng(seed, seed*2+1);

    // ---- Scene (hardcoded Cornell-like)
    HittableList objs;
    auto white = make_shared<Lambertian>(Vec3(0.75,0.75,0.75));
    auto red   = make_shared<Lambertian>(Vec3(0.75,0.15,0.15));
    auto green = make_shared<Lambertian>(Vec3(0.15,0.75,0.15));
    auto steel = make_shared<Metal>(Vec3(0.75,0.75,0.75), 0.05);
    auto glass = make_shared<Dielectric>(1.5);
    auto light = make_shared<DiffuseLight>(Vec3(1.0,0.97,0.92), 8000.0);

    auto f0 = make_shared<XZRect>(-1,1,-2.2,0.2,0, white);  f0->id=10; objs.add(f0);
    auto ceil = make_shared<XZRect>(-1,1,-2.2,0.2,2, white); ceil->id=11; objs.add(ceil);
    auto back = make_shared<XYRect>(-1,1,0,2,-2.2, white);  back->id=12; objs.add(back);
    auto left = make_shared<YZRect>(0,2,-2.2,0.2,-1, red);  left->id=13; objs.add(left);
    auto rgt  = make_shared<YZRect>(0,2,-2.2,0.2, 1, green);rgt->id=14; objs.add(rgt);

    auto rect_light = make_shared<XZRect>(-0.4,0.4,-1.0,-0.2,1.95, light); rect_light->id=99; objs.add(rect_light);
    auto s1 = make_shared<Sphere>(Vec3(-0.4,0.35,-1.4), 0.35, glass); s1->id=1; objs.add(s1);
    auto s2 = make_shared<Sphere>(Vec3( 0.5,0.50,-1.0), 0.50, steel); s2->id=2; objs.add(s2);

    // BVH
    vector<shared_ptr<Hittable>> v = objs.objects;
    BVHNode world(v, 0, (int)v.size());

    // Camera
    double aspect = double(W)/double(H);
    Camera cam(Vec3(0,1,1.2), Vec3(0,1,-1.1), Vec3(0,1,0),
               50.0, aspect, 0.0, 1.0, 0.0, 1.0);

    // Buffers
    AOV aov;
    aov.beauty.resize(W*H);
    aov.albedo.resize(W*H, Vec3(0,0,0));
    aov.normal.resize(W*H, Vec3(0,0,0));
    aov.depth.resize(W*H, 0.0f);
    aov.inst_id.resize(W*H, 0);

    auto rand01 = [&](){ return rng.uniform_double(); };

    // Render
    for(int y=H-1; y>=0; --y){
        for(int x=0; x<W; ++x){
            Vec3 col(0,0,0); FirstHit fh;
            for(int s=0; s<spp; ++s){
                double u = (x + rand01())/(W-1);
                double v = (y + rand01())/(H-1);
                col += ray_color(cam.get_ray(u,v), world, max_depth, (s==0?&fh:nullptr));
            }
            col /= double(spp);
            int idx = (H-1-y)*W + x;
            aov.beauty[idx] = col;
            if (fh.filled){
                aov.albedo[idx] = fh.albedo;
                aov.normal[idx] = fh.normal;
                aov.depth [idx] = fh.depth;
                aov.inst_id[idx]= fh.id;
            }
        }
    }

    // Write linear EXRs (beauty/albedo/normal/depth). Mask next step.
    std::string p_rgb   = outdir + "/rgb.exr";
    std::string p_alb   = outdir + "/albedo.exr";
    std::string p_nrm   = outdir + "/normal.exr";
    std::string p_depth = outdir + "/depth.exr";
    vf::write_rgb_exr(p_rgb.c_str(), W, H, aov.beauty);
    vf::write_rgb_exr(p_alb.c_str(), W, H, aov.albedo);
    vf::write_rgb_exr(p_nrm.c_str(), W, H, aov.normal);
    vf::write_float_exr(p_depth.c_str(), W, H, aov.depth, "Z");

    // Also write a quick tonemapped preview PPM
    std::ofstream ppm((outdir + "/rgb.ppm").c_str(), std::ios::binary);
    ppm << "P6\n" << W << " " << H << "\n255\n";
    for(auto& c : aov.beauty){
        Vec3 m = aces_tonemap(c);
        m = {std::sqrt(m.x), std::sqrt(m.y), std::sqrt(m.z)};
        unsigned char r = (unsigned char)(256*clamp01(m.x));
        unsigned char g = (unsigned char)(256*clamp01(m.y));
        unsigned char b = (unsigned char)(256*clamp01(m.z));
        ppm.write((char*)&r,1); ppm.write((char*)&g,1); ppm.write((char*)&b,1);
    }
    ppm.close();

    return 0;
}
