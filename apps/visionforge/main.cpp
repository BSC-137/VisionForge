#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>
#include <vector>
#include <filesystem>  // NEW: to create ../out

#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/hittable_list.hpp"
#include "visionforge/sphere.hpp"
#include "visionforge/xz_rect.hpp"
#include "visionforge/camera.hpp"
#include "visionforge/material.hpp"
#include "visionforge/lambertian.hpp"
#include "visionforge/metal.hpp"
#include "visionforge/dielectric.hpp"
#include "visionforge/diffuse_light.hpp"
#include "visionforge/onb.hpp"
#include "visionforge/xy_rect.hpp"
#include "visionforge/yz_rect.hpp"
#include "visionforge/bvh.hpp"

// ---------------- RENDER CONFIG ----------------
static const bool PREVIEW = false; // false = final render

static const int  SPP_PREVIEW = 16;
static const int  SPP_FINAL   = 100;

static const int  MAX_DEPTH_PREVIEW = 8;
static const int  MAX_DEPTH_FINAL   = 25;

static const int  LIGHT_SAMPLES_PREVIEW = 2;
static const int  LIGHT_SAMPLES_FINAL   = 8;

static const int  BRDF_SAMPLES_PER_HIT  = 1;

static const double F_NUMBER = 2.0;
static const double SHUTTER  = 1.0/30;
static const int    ISO      = 400;
static const double EXPOSURE_COMP = 8.0;
// ------------------------------------------------

static inline double clamp01(double x){ return x<0 ? 0 : (x>1 ? 1 : x); }

static inline double exposure_scale(double fnum, double shutter_s, int iso){
    double EV100 = std::log2((fnum*fnum)/shutter_s);
    return 0.18 * std::pow(2.0, -EV100) * (100.0/double(iso));
}

static inline Vec3 aces_tonemap(const Vec3& c){
    const double a=2.51,b=0.03,c2=2.43,d=0.59,e=0.14;
    auto tm=[&](double x){ double num=x*(a*x+b), den=x*(c2*x+d)+e; return clamp01(num/den); };
    return Vec3(tm(c.x), tm(c.y), tm(c.z));
}

static inline bool get_lambert_albedo(const std::shared_ptr<Material>& m, Vec3& out_albedo){
    auto* lam = dynamic_cast<Lambertian*>(m.get());
    if (!lam) return false;
    out_albedo = lam->albedo;
    return true;
}

static bool rect_pdf_omega_from_dir(const XZRect& rect, const Vec3& p, const Vec3& wi,
                                    double& pdf_omega, double& cos_l, double& dist)
{
    if (std::fabs(wi.y) < 1e-8) return false;
    double t = (rect.k - p.y) / wi.y;
    if (t <= 0.001) return false;

    Vec3 hit = p + wi * t;
    if (hit.x < rect.x0 || hit.x > rect.x1 || hit.z < rect.z0 || hit.z > rect.z1) return false;

    dist = t;
    cos_l = std::max(0.0, dot(rect.light_normal(), -wi));
    if (cos_l <= 0.0) return false;

    double A = rect.area();
    double dist2 = dist * dist;
    pdf_omega = dist2 / (cos_l * A);
    return pdf_omega > 1e-12;
}

Vec3 ray_color(const Ray& r, const Hittable& world, const XZRect* area_light, int depth, int max_depth){
    if (depth <= 0) return Vec3(0,0,0);

    HitRecord rec;
    if (!world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        return Vec3(0,0,0);
    }

    Vec3 emitted = rec.mat->emitted(rec);

    Ray scattered;
    Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered)) {
        return emitted;
    }

    // Russian roulette termination
    if (depth < max_depth - 4) {
        double p = std::max(attenuation.x, std::max(attenuation.y, attenuation.z));
        p = clamp01(p);
        if (p < 0.05) p = 0.05;
        if (random_double() > p) return emitted;
        attenuation /= p;
    }

    Vec3 indirect = attenuation * ray_color(scattered, world, area_light, depth - 1, max_depth);

    Vec3 direct(0,0,0);
    Vec3 albedo;
    const int LIGHT_SAMPLES_PER_HIT = PREVIEW ? LIGHT_SAMPLES_PREVIEW : LIGHT_SAMPLES_FINAL;

    // Direct light sampling
    if (area_light && get_lambert_albedo(rec.mat, albedo)) {
        Vec3 L_light(0,0,0);
        for (int s=0; s<LIGHT_SAMPLES_PER_HIT; ++s) {
            Vec3 lp = area_light->sample_point();
            Vec3 toL = lp - rec.point;
            double dist2 = dot(toL, toL);
            if (dist2 <= 1e-12) continue;

            double dist = std::sqrt(dist2);
            Vec3 wi = toL / dist;

            double cos_i = std::max(0.0, dot(rec.normal, wi));
            double cos_l = std::max(0.0, dot(area_light->light_normal(), -wi));
            if (cos_i <= 0.0 || cos_l <= 0.0) continue;

            Ray shadow_ray(rec.point, wi);
            HitRecord shadow_hit;
            if (world.hit(shadow_ray, 0.001, dist - 0.001, shadow_hit)) continue;

            double A = area_light->area();
            double pdf_light = dist2 / (cos_l * A);
            double pdf_brdf  = cos_i / PI;
            double w = pdf_light / (pdf_light + pdf_brdf);

            Vec3 Le = area_light->mat->emitted(rec);
            Vec3 f  = (albedo / PI);
            L_light += w * Le * f * (cos_i / pdf_light);
        }
        L_light /= double(LIGHT_SAMPLES_PER_HIT);

        direct = L_light;
    }

    return emitted + direct + indirect;
}

int main(){
    srand((unsigned)time(0));

    const int width  = 640;
    const int height = 360;
    const int samples_per_pixel = PREVIEW ? SPP_PREVIEW : SPP_FINAL;
    const int max_depth         = PREVIEW ? MAX_DEPTH_PREVIEW : MAX_DEPTH_FINAL;
    const double aspect = double(width) / double(height);

    // NEW: ensure ../out exists and write the image there
    std::filesystem::create_directories("../out");
    std::ofstream file("../out/image.ppm", std::ios::binary);

    file << "P6\n" << width << " " << height << "\n255\n";
    std::vector<unsigned char> out;
    out.reserve(width * height * 3);

    // Camera
    Vec3 lookfrom(0.0, 1.0, 1.2);
    Vec3 lookat  (0.0, 1.0, -1.10);
    double focus_dist = (lookfrom - lookat).length();
    double aperture   = 0.0;

    double time0 = 0;
    double time1 = 1;

    Camera cam(lookfrom, lookat, Vec3(0,1,0),
            50.0,
            aspect,
            aperture, focus_dist,
            time0, time1);

    // Scene
    const double room_min_x = -1.0, room_max_x =  1.0;
    const double room_min_y =  0.0, room_max_y =  2.0;
    const double room_min_z = -2.2, room_max_z =  0.2;

    HittableList objects;

    auto white   = std::make_shared<Lambertian>(Vec3(0.75, 0.75, 0.75));
    auto red     = std::make_shared<Lambertian>(Vec3(0.75, 0.15, 0.15));
    auto green   = std::make_shared<Lambertian>(Vec3(0.15, 0.75, 0.15));
    auto steel   = std::make_shared<Metal>(Vec3(0.75, 0.75, 0.75), 0.05);
    auto glass   = std::make_shared<Dielectric>(1.5);
    auto light_mat = std::make_shared<DiffuseLight>(Vec3(1.0, 0.97, 0.92), 8000.0);

    objects.add(std::make_shared<XZRect>(room_min_x, room_max_x, room_min_z, room_max_z, room_min_y, white));
    objects.add(std::make_shared<XZRect>(room_min_x, room_max_x, room_min_z, room_max_z, room_max_y, white));

    objects.add(std::make_shared<XYRect>(room_min_x, room_max_x, room_min_y, room_max_y, room_min_z, white));
    objects.add(std::make_shared<YZRect>(room_min_y, room_max_y, room_min_z, room_max_z, room_min_x, red));
    objects.add(std::make_shared<YZRect>(room_min_y, room_max_y, room_min_z, room_max_z, room_max_x, green));

    const double Lx0 = -0.4, Lx1 = 0.4;
    const double Lz0 = -1.0, Lz1 = -0.2;
    const double Ly  = 1.95;
    auto rect_light = std::make_shared<XZRect>(Lx0, Lx1, Lz0, Lz1, Ly, light_mat);
    objects.add(rect_light);

    objects.add(std::make_shared<Sphere>(Vec3(-0.4, 0.35, -1.4), 0.35, glass));
    objects.add(std::make_shared<Sphere>(Vec3( 0.5, 0.50, -1.0), 0.50, steel));

    // BVH build
    std::vector<std::shared_ptr<Hittable>> objs_vec = objects.objects;
    BVHNode world(objs_vec, 0, objs_vec.size());

    double exposure = exposure_scale(F_NUMBER, SHUTTER, ISO) * EXPOSURE_COMP;

    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            Vec3 pixel(0,0,0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                double u = (i + random_double()) / (width  - 1);
                double v = (j + random_double()) / (height - 1);
                Ray r = cam.get_ray(u, v);
                pixel += ray_color(r, world, rect_light.get(), max_depth, max_depth);
            }
            pixel /= double(samples_per_pixel);
            pixel *= exposure;
            Vec3 mapped = aces_tonemap(pixel);
            mapped = Vec3(std::sqrt(mapped.x), std::sqrt(mapped.y), std::sqrt(mapped.z));

            out.push_back((unsigned char)(256 * clamp01(mapped.x)));
            out.push_back((unsigned char)(256 * clamp01(mapped.y)));
            out.push_back((unsigned char)(256 * clamp01(mapped.z)));
        }
    }

    file.write(reinterpret_cast<const char*>(out.data()), out.size());
    file.close();

    std::cout << "Wrote ../out/image.ppm\n";
    return 0;
}
