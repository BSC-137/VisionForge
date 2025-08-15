// ===== VisionForge — Procedural Desert (balanced dunes, fast) =====
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <random>

#include "visionforge/vec3.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/hittable_list.hpp"
#include "visionforge/xz_rect.hpp"
#include "visionforge/yz_rect.hpp"
#include "visionforge/xy_rect.hpp"
#include "visionforge/camera.hpp"
#include "visionforge/material.hpp"
#include "visionforge/lambertian.hpp"
#include "visionforge/diffuse_light.hpp"
#include "visionforge/bvh.hpp"
#include "visionforge/perlin.hpp"
#include "visionforge/aabb.hpp"

// ---------------- RENDER CONFIG ----------------
static const bool PREVIEW = false;

static const int  WIDTH   = 960;
static const int  HEIGHT  = 540;

static const int  SPP_PREVIEW = 12;
static const int  SPP_FINAL   = 60;

static const int  MAX_DEPTH_PREVIEW = 6;
static const int  MAX_DEPTH_FINAL   = 12;

static const int  LIGHT_SAMPLES_PREVIEW = 1;
static const int  LIGHT_SAMPLES_FINAL   = 1; // 1 is enough for a “sun”

// Adaptive sampling
static const int    MIN_SPP = 6;
static const double REL_NOISE_TARGET = 0.02; // 2% luminance CI

// Exposure
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
static inline double luminance(const Vec3& c){
    return 0.2126*c.x + 0.7152*c.y + 0.0722*c.z;
}
static inline Vec3 sky(const Vec3& dir){
    double t = 0.5 * (dir.y + 1.0);
    Vec3 horizon(0.92, 0.88, 0.80);
    Vec3 zenith (0.60, 0.74, 0.98);
    return (1.0 - t) * horizon + t * zenith;
}
static inline bool get_lambert_albedo(const std::shared_ptr<Material>& m, Vec3& out_albedo){
    auto* lam = dynamic_cast<Lambertian*>(m.get());
    if (!lam) return false;
    out_albedo = lam->albedo;
    return true;
}

// ---------- Triangle (per-vertex normals) ----------
class Triangle : public Hittable {
public:
    Vec3 p0, p1, p2, n0, n1, n2;
    std::shared_ptr<Material> mat;
    AABB box;

    Triangle()=default;
    Triangle(const Vec3& a,const Vec3& b,const Vec3& c,
             const Vec3& na,const Vec3& nb,const Vec3& nc,
             std::shared_ptr<Material> m)
        : p0(a),p1(b),p2(c),n0(normalize(na)),n1(normalize(nb)),n2(normalize(nc)),mat(std::move(m))
    {
        const double eps=1e-6;
        Vec3 mn(std::fmin(p0.x,std::fmin(p1.x,p2.x)),
                std::fmin(p0.y,std::fmin(p1.y,p2.y)),
                std::fmin(p0.z,std::fmin(p1.z,p2.z)));
        Vec3 mx(std::fmax(p0.x,std::fmax(p1.x,p2.x)),
                std::fmax(p0.y,std::fmax(p1.y,p2.y)),
                std::fmax(p0.z,std::fmax(p1.z,p2.z)));
        box = AABB(mn-Vec3(eps,eps,eps), mx+Vec3(eps,eps,eps));
    }
    bool hit(const Ray& r,double tmin,double tmax,HitRecord& rec) const override{
        const Vec3 e1=p1-p0, e2=p2-p0;
        const Vec3 pvec=cross(r.direction,e2);
        const double det=dot(e1,pvec);
        if (std::fabs(det)<1e-10) return false;
        const double invDet=1.0/det;
        const Vec3 tvec=r.origin-p0;
        const double u=dot(tvec,pvec)*invDet; if(u<0||u>1) return false;
        const Vec3 qvec=cross(tvec,e1);
        const double v=dot(r.direction,qvec)*invDet; if(v<0||u+v>1) return false;
        const double t=dot(e2,qvec)*invDet; if(t<tmin||t>tmax) return false;

        rec.t=t; rec.point=r.origin+r.direction*t;
        const double w=1.0-u-v;
        Vec3 n=normalize(w*n0+u*n1+v*n2);
        if (dot(n,r.direction)>0) n=-n;
        rec.normal=n;
        rec.mat=mat;
        return true;
    }
    bool bounding_box(AABB& out_box) const override{ out_box=box; return true; }
};

// ---------- HeightField (balanced ridged + domain-warped) ----------
struct HeightField {
    double xmin,xmax,zmin,zmax; int nx,nz; double amp,scale; vf::perlin noise;
    std::vector<double> h;
    HeightField(double X0,double X1,double Z0,double Z1,int NX,int NZ,double A,double S,uint32_t seed=1337)
        : xmin(X0),xmax(X1),zmin(Z0),zmax(Z1),nx(NX),nz(NZ),amp(A),scale(S),noise(seed),h((NX+1)*(NZ+1)){}
    inline int idx(int i,int k) const { return k*(nx+1)+i; }

    void generate(){
        auto ridged_fbm=[&](const Vec3& p){
            double a=1.0,f=1.0,s=0.0,nrm=0.0;
            for(int o=0;o<5;++o){
                double n = 1.0 - std::fabs(noise.fbm(p*f, 1, 0.5, 2.0)); // ridge in [0,1]
                n = 0.85*n + 0.15*n*n;  // soften crests slightly
                s += a * n; nrm += a;
                a *= 0.62;             // smoother falloff (less jagged)
                f *= 1.9;              // slightly lower lacunarity
            }
            return s/std::max(1e-8,nrm);
        };
        auto warp=[&](const Vec3& p){
            return p + 0.35*Vec3(
                noise.fbm(p*1.7,3,0.5,2.1),
                noise.fbm(p*2.0,3,0.5,2.0),
                noise.fbm(p*1.8,3,0.5,2.2));
        };
        for (int k=0;k<=nz;++k){
            double z=zmin+(zmax-zmin)*(double(k)/nz);
            for (int i=0;i<=nx;++i){
                double x=xmin+(xmax-xmin)*(double(i)/nx);
                Vec3 pd=warp(Vec3(x*scale,0.0,z*scale));
                double dunes = ridged_fbm(pd);                     // [0,1]
                double mid   = 0.25 * noise.fbm(pd*2.4,4,0.5,2.0); // ~[-0.25,0.25]
                double fine  = 0.06 * noise.fbm(pd*10.0,3,0.5,2.0);
                h[idx(i,k)]  = amp * ((1.1*dunes - 0.55) + mid + fine);
            }
        }
    }
    double height_at(double x,double z) const {
        double u=(x-xmin)/(xmax-xmin), v=(z-zmin)/(zmax-zmin);
        u=std::clamp(u,0.0,1.0); v=std::clamp(v,0.0,1.0);
        double fx=u*nx, fz=v*nz; int i=std::clamp((int)std::floor(fx),0,nx-1);
        int k=std::clamp((int)std::floor(fz),0,nz-1); double du=fx-i, dv=fz-k;
        auto H=[&](int ii,int kk){ return h[idx(ii,kk)]; };
        double h00=H(i,k), h10=H(i+1,k), h01=H(i,k+1), h11=H(i+1,k+1);
        double h0=h00*(1-du)+h10*du, h1=h01*(1-du)+h11*du; return h0*(1-dv)+h1*dv;
    }
    Vec3 normal_at(int i,int k) const {
        auto H=[&](int ii,int kk){ ii=std::clamp(ii,0,nx); kk=std::clamp(kk,0,nz); return h[idx(ii,kk)]; };
        double hx=(H(i+1,k)-H(i-1,k))*0.5, hz=(H(i,k+1)-H(i,k-1))*0.5;
        double dx=(xmax-xmin)/nx, dz=(zmax-zmin)/nz;
        Vec3 tx(dx,hx,0), tz(0,hz,dz); return normalize(cross(tz,tx));
    }
    void to_triangles(HittableList& out, const std::shared_ptr<Material>& m) const {
        for(int k=0;k<nz;++k){
            double z0=zmin+(zmax-zmin)*(double(k)/nz);
            double z1=z0 + (zmax-zmin)/nz;
            for(int i=0;i<nx;++i){
                double x0=xmin+(xmax-xmin)*(double(i)/nx);
                double x1=x0 + (xmax-xmin)/nx;
                Vec3 v00(x0,h[idx(i  ,k  )],z0), v10(x1,h[idx(i+1,k  )],z0);
                Vec3 v01(x0,h[idx(i  ,k+1)],z1), v11(x1,h[idx(i+1,k+1)],z1);
                Vec3 n00=normal_at(i  ,k  ), n10=normal_at(i+1,k  ),
                     n01=normal_at(i  ,k+1), n11=normal_at(i+1,k+1);
                if (((i^k)&1)==0){
                    out.add(std::make_shared<Triangle>(v00,v10,v11,n00,n10,n11,m));
                    out.add(std::make_shared<Triangle>(v00,v11,v01,n00,n11,n01,m));
                } else {
                    out.add(std::make_shared<Triangle>(v00,v10,v01,n00,n10,n01,m));
                    out.add(std::make_shared<Triangle>(v10,v11,v01,n10,n11,n01,m));
                }
            }
        }
    }
};

// ---------- Micro bump for sand (toned down) ----------
static vf::perlin g_bump(424242);
static const double BUMP_FREQ  = 5.0;  // slightly coarser
static const double BUMP_SCALE = 0.22; // subtler
static std::shared_ptr<Lambertian> sand_ptr; // for identity check

// ---------- MIS-style integrator, templated light rect ----------
template<typename RectT>
Vec3 ray_color(const Ray& r, const Hittable& world, const RectT* area_light, int depth, int max_depth){
    if (depth <= 0) return Vec3(0,0,0);

    HitRecord rec;
    if (!world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        return sky(normalize(r.direction));
    }

    // Micro-normal perturb just for sand
    if (rec.mat == sand_ptr) {
        Vec3 g = g_bump.grad(rec.point * BUMP_FREQ);
        // (your Vec3 may not have -=)
        g = g - rec.normal * dot(g, rec.normal);
        rec.normal = normalize(rec.normal + BUMP_SCALE * g);
    }

    Vec3 emitted = rec.mat->emitted(rec);

    Ray scattered;
    Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered)) {
        return emitted;
    }

    // Early cut for tiny throughput
    if (attenuation.x < 1e-3 && attenuation.y < 1e-3 && attenuation.z < 1e-3)
        return emitted;

    // Russian roulette
    if (depth < max_depth - 4) {
        double p = std::max(attenuation.x, std::max(attenuation.y, attenuation.z));
        p = clamp01(p);
        if (p < 0.05) p = 0.05;
        if (random_double() > p) return emitted;
        attenuation /= p;
    }

    Vec3 indirect = attenuation * ray_color(scattered, world, area_light, depth - 1, max_depth);

    // Direct light (area rect)
    Vec3 direct(0,0,0);
    Vec3 albedo;
    const int LIGHT_SAMPLES_PER_HIT = PREVIEW ? LIGHT_SAMPLES_PREVIEW : LIGHT_SAMPLES_FINAL;

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

// ---------- Utility: add a cube from 6 rects ----------
static void add_cube(HittableList& list, const Vec3& c, double edge, const std::shared_ptr<Material>& m) {
    double h = edge * 0.5;
    double x0 = c.x - h, x1 = c.x + h;
    double y0 = c.y - h, y1 = c.y + h;
    double z0 = c.z - h, z1 = c.z + h;
    list.add(std::make_shared<XZRect>(x0, x1, z0, z1, y1, m));
    list.add(std::make_shared<XZRect>(x0, x1, z0, z1, y0, m));
    list.add(std::make_shared<XYRect>(x0, x1, y0, y1, z1, m));
    list.add(std::make_shared<XYRect>(x0, x1, y0, y1, z0, m));
    list.add(std::make_shared<YZRect>(y0, y1, z0, z1, x0, m));
    list.add(std::make_shared<YZRect>(y0, y1, z0, z1, x1, m));
}

int main(){
    srand((unsigned)time(0));
    std::filesystem::create_directories("../out");

    const int width  = WIDTH;
    const int height = HEIGHT;
    const int spp_target = PREVIEW ? SPP_PREVIEW : SPP_FINAL;
    const int max_depth  = PREVIEW ? MAX_DEPTH_PREVIEW : MAX_DEPTH_FINAL;
    const double aspect  = double(width)/double(height);

    // Camera
    Vec3 lookfrom(18.0, 8.0, 24.0);
    Vec3 lookat  ( 0.0, 1.2,  0.0);
    double focus_dist = (lookfrom - lookat).length();
    Camera cam(lookfrom, lookat, Vec3(0,1,0), 35.0, aspect, 0.0, focus_dist, 0.0, 1.0);

    // Materials
    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    auto sand   = sand_ptr;
    auto red_m  = std::make_shared<Lambertian>(Vec3(0.92, 0.18, 0.14));
    auto blu_m  = std::make_shared<Lambertian>(Vec3(0.14, 0.55, 0.92));
    auto grn_m  = std::make_shared<Lambertian>(Vec3(0.18, 0.84, 0.22));
    auto wht_m  = std::make_shared<Lambertian>(Vec3(0.85, 0.85, 0.85));
    auto sun    = std::make_shared<DiffuseLight>(Vec3(1.0, 0.98, 0.92), 12000.0);

    // Terrain (balanced)
    const double XMIN=-22, XMAX=22, ZMIN=-22, ZMAX=22;
    const int NX = 96, NZ = 96; // ~18k tris → fast
    HeightField hf(XMIN, XMAX, ZMIN, ZMAX, NX, NZ, /*amp=*/1.8, /*scale=*/0.14, /*seed=*/1337);
    hf.generate();

    HittableList objects;
    hf.to_triangles(objects, sand);

    // Side "sun" as a YZ rect at x=-30, pointing +X (casts relief shadows)
    auto rect_light = std::make_shared<YZRect>(
        2.0, 14.0,       // y range (slightly narrowed to soften contrast)
       -25.0, 25.0,      // z range
       -30.0,            // x = -30 (normal +X in your YZRect)
       sun);
    objects.add(rect_light);

    // Random cubes (red/blue/green/white)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> px(XMIN+2.0, XMAX-2.0);
    std::uniform_real_distribution<double> pz(ZMIN+2.0, ZMAX-2.0);
    std::uniform_int_distribution<int> col(0,3);
    double edge = 1.3;
    for (int i=0;i<5;++i){
        double x = px(rng), z = pz(rng);
        double y = hf.height_at(x,z) + edge*0.5;
        std::shared_ptr<Material> m = (col(rng)==0)?red_m: (col(rng)==1)?blu_m: (col(rng)==2)?grn_m:wht_m;
        add_cube(objects, Vec3(x,y,z), edge, m);
    }

    // BVH
    std::vector<std::shared_ptr<Hittable>> objs_vec = objects.objects;
    BVHNode world(objs_vec, 0, objs_vec.size());

    // Framebuffer
    std::vector<unsigned char> fb(width * height * 3, 0);
    double exposure = exposure_scale(F_NUMBER, SHUTTER, ISO) * EXPOSURE_COMP;

    // Render
    #if defined(VF_USE_OMP)
      #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            Vec3 sum(0,0,0);
            double meanL = 0.0, M2 = 0.0;
            int s = 0;
            for (; s < spp_target; ++s) {
                double u = (i + random_double()) / (width  - 1);
                double v = (j + random_double()) / (height - 1);
                Ray r = cam.get_ray(u, v);
                Vec3 c = ray_color(r, world, rect_light.get(), max_depth, max_depth);
                sum += c;

                // adaptive stop
                double L = luminance(c);
                double delta = L - meanL; meanL += delta / double(s+1); M2 += delta * (L - meanL);
                if (s+1 >= MIN_SPP) {
                    double var = (s > 0) ? (M2 / s) : 0.0;
                    double sigma = std::sqrt(std::max(0.0, var));
                    double ci95  = 1.96 * sigma / std::sqrt(double(s+1));
                    if (ci95 < REL_NOISE_TARGET * std::max(1e-3, meanL)) break;
                }
            }
            Vec3 pixel = (sum / double(s+1)) * exposure;
            Vec3 mapped = aces_tonemap(pixel);
            mapped = Vec3(std::sqrt(mapped.x), std::sqrt(mapped.y), std::sqrt(mapped.z));
            int idx = ((height-1-j)*width + i)*3;
            fb[idx+0] = (unsigned char)(256 * clamp01(mapped.x));
            fb[idx+1] = (unsigned char)(256 * clamp01(mapped.y));
            fb[idx+2] = (unsigned char)(256 * clamp01(mapped.z));
        }
    }

    std::ofstream file("../out/image.ppm", std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(fb.data()), fb.size());
    std::cout << "Wrote ../out/image.ppm\n";
    return 0;
}
