// ===== VisionForge — Procedural Desert (dunes + RGBW cubes with tight bboxes) =====
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
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

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

#include "visionforge/sky.hpp"
#include "visionforge/bbox.hpp"

// ------------ CONFIG ------------
static const bool   PREVIEW = false;
static const int    WIDTH   = 1280;
static const int    HEIGHT  = 720;

static const int  SPP_PREVIEW = 12;
static const int  SPP_FINAL   = 96;
static const int  MAX_DEPTH_PREVIEW = 6;
static const int  MAX_DEPTH_FINAL   = 16;

static const int  LIGHT_SAMPLES_PREVIEW = 1;
static const int  LIGHT_SAMPLES_FINAL   = 1;

// Adaptive sampling
static const int    MIN_SPP = 6;
static const double REL_NOISE_TARGET = 0.020;

// Exposure
static const double F_NUMBER = 2.0;
static const double SHUTTER  = 1.0/30.0;
static const int    ISO      = 400;
static const double EXPOSURE_COMP = 6.5;

#ifndef PI
#define PI 3.14159265358979323846
#endif

static inline double clamp01(double x){ return x<0 ? 0 : (x>1 ? 1 : x); }
static inline Vec3 clamp01(const Vec3& c){ return Vec3(clamp01(c.x),clamp01(c.y),clamp01(c.z)); }
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

// ====================== Transform wrappers ======================
class Translate : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    Vec3 offset;
    Translate(std::shared_ptr<Hittable> p, const Vec3& d) : ptr(std::move(p)), offset(d) {}
    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        Ray moved(r.origin - offset, r.direction, r.time);
        if (!ptr->hit(moved, t_min, t_max, rec)) return false;
        rec.point += offset;
        return true;
    }
    bool bounding_box(AABB& out_box) const override {
        AABB b; if (!ptr->bounding_box(b)) return false;
        out_box = AABB(b.min()+offset, b.max()+offset);
        return true;
    }
};

class RotateX : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double sin_t, cos_t;
    AABB box;
    RotateX(std::shared_ptr<Hittable> p, double angle_deg) : ptr(std::move(p)) {
        double rad = angle_deg * PI/180.0;
        sin_t = std::sin(rad); cos_t = std::cos(rad);
        AABB b;
        if (ptr->bounding_box(b)) {
            Vec3 minv( 1e9, 1e9, 1e9), maxv(-1e9,-1e9,-1e9);
            for (int i=0;i<2;i++) for(int j=0;j<2;j++) for(int k=0;k<2;k++){
                double x = i? b.max().x : b.min().x;
                double y = j? b.max().y : b.min().y;
                double z = k? b.max().z : b.min().z;
                double ny =  cos_t*y - sin_t*z;
                double nz =  sin_t*y + cos_t*z;
                Vec3 t(x,ny,nz);
                minv = min_vec(minv, t);
                maxv = max_vec(maxv, t);
            }
            box = AABB(minv,maxv);
        }
    }
    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        Vec3 o = r.origin, d = r.direction;
        o.y =  cos_t*r.origin.y + sin_t*r.origin.z;
        o.z = -sin_t*r.origin.y + cos_t*r.origin.z;
        d.y =  cos_t*r.direction.y + sin_t*r.direction.z;
        d.z = -sin_t*r.direction.y + cos_t*r.direction.z;
        Ray rr(o,d,r.time);
        if (!ptr->hit(rr,tmin,tmax,rec)) return false;
        double y =  cos_t*rec.point.y - sin_t*rec.point.z;
        double z =  sin_t*rec.point.y + cos_t*rec.point.z;
        rec.point.y = y; rec.point.z = z;
        double ny =  cos_t*rec.normal.y - sin_t*rec.normal.z;
        double nz =  sin_t*rec.normal.y + cos_t*rec.normal.z;
        rec.normal.y = ny; rec.normal.z = nz;
        return true;
    }
    bool bounding_box(AABB& out_box) const override { out_box = box; return true; }
};

class RotateZ : public Hittable {
public:
    std::shared_ptr<Hittable> ptr;
    double sin_t, cos_t;
    AABB box;
    RotateZ(std::shared_ptr<Hittable> p, double angle_deg) : ptr(std::move(p)) {
        double rad = angle_deg * PI/180.0;
        sin_t = std::sin(rad); cos_t = std::cos(rad);
        AABB b;
        if (ptr->bounding_box(b)) {
            Vec3 minv( 1e9, 1e9, 1e9), maxv(-1e9,-1e9,-1e9);
            for (int i=0;i<2;i++) for(int j=0;j<2;j++) for(int k=0;k<2;k++){
                double x = i? b.max().x : b.min().x;
                double y = j? b.max().y : b.min().y;
                double z = k? b.max().z : b.min().z;
                double nx =  cos_t*x - sin_t*y;
                double ny =  sin_t*x + cos_t*y;
                Vec3 t(nx,ny,z);
                minv = min_vec(minv, t);
                maxv = max_vec(maxv, t);
            }
            box = AABB(minv,maxv);
        }
    }
    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        Vec3 o = r.origin, d = r.direction;
        o.x =  cos_t*r.origin.x + sin_t*r.origin.y;
        o.y = -sin_t*r.origin.x + cos_t*r.origin.y;
        d.x =  cos_t*r.direction.x + sin_t*r.direction.y;
        d.y = -sin_t*r.direction.x + cos_t*r.direction.y;
        Ray rr(o,d,r.time);
        if (!ptr->hit(rr,tmin,tmax,rec)) return false;
        double x =  cos_t*rec.point.x - sin_t*rec.point.y;
        double y =  sin_t*rec.point.x + cos_t*rec.point.y;
        rec.point.x = x; rec.point.y = y;
        double nx =  cos_t*rec.normal.x - sin_t*rec.normal.y;
        double ny =  sin_t*rec.normal.x + cos_t*rec.normal.y;
        rec.normal.x = nx; rec.normal.y = ny;
        return true;
    }
    bool bounding_box(AABB& out_box) const override { out_box = box; return true; }
};

// ====================== Triangle mesh piece (per-vertex normals) ======================
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

// ====================== HeightField (dunes) ======================
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
                double n = 1.0 - std::fabs(noise.fbm(p*f, 1, 0.5, 2.0));
                n = 0.85*n + 0.15*n*n;
                s += a * n; nrm += a;
                a *= 0.62; f *= 1.9;
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
                double dunes = ridged_fbm(pd);
                double mid   = 0.25 * noise.fbm(pd*2.4,4,0.5,2.0);
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
        double h0=h00*(1-du)+h10*du;
        double h1=h01*(1-du)+h11*du;
        return h0*(1-dv)+h1*dv;
    }
    Vec3 normal_at(int i,int k) const {
        auto H=[&](int ii,int kk){ ii=std::clamp(ii,0,nx); kk=std::clamp(kk,0,nz); return h[idx(ii,kk)]; };
        double hx=(H(i+1,k)-H(i-1,k))*0.5, hz=(H(i,k+1)-H(i,k-1))*0.5;
        double dx=(xmax-xmin)/nx, dz=(zmax-zmin)/nz;
        Vec3 tx(dx,hx,0), tz(0,hz,dz); return normalize(cross(tz,tx));
    }

    // Carve a shallow footprint + rim at (x0,z0)
    void carve_footprint(double x0, double z0, double edge, double sink, double rim_amp){
        double r = 0.6 * edge;
        double r2 = r*r;
        double r_out = r*1.8;
        for (int k=0;k<=nz;++k){
            double z = zmin + (zmax-zmin)*(double(k)/nz);
            for (int i=0;i<=nx;++i){
                double x = xmin + (xmax-xmin)*(double(i)/nx);
                double dx=x-x0, dz=z-z0; double d2=dx*dx+dz*dz;
                if (d2 > r_out*r_out) continue;

                double depress = sink * std::exp(-(d2)/(2.0*r2));
                double ring = std::exp(-std::pow((std::sqrt(d2)/r - 1.1)/0.18, 2.0));
                double berm = rim_amp * ring;

                h[idx(i,k)] -= depress;
                h[idx(i,k)] += berm;
            }
        }
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

// ====================== Sand micro bump ======================
static vf::perlin g_bump(424242);
static const double BUMP_FREQ  = 5.0;
static const double BUMP_SCALE = 0.22;
static std::shared_ptr<Lambertian> sand_ptr; // identity check

// ====================== Sky model ======================
static Sky g_sky(/*sun_az_deg=*/300.0, /*sun_elev_deg=*/12.0, /*turbidity=*/3.5);

// Helper: expose albedo if the material is Lambertian
static inline bool get_lambert_albedo(const std::shared_ptr<Material>& m, Vec3& out_albedo) {
    if (auto* lam = dynamic_cast<Lambertian*>(m.get())) {
        out_albedo = lam->albedo;
        return true;
    }
    return false;
}

// ====================== Integrator (with light MIS) ======================
template<typename RectT>
Vec3 ray_color(const Ray& r, const Hittable& world, const RectT* area_light, int depth, int max_depth){
    if (depth <= 0) return Vec3(0,0,0);
    HitRecord rec;
    if (!world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        return g_sky.eval(normalize(r.direction));
    }

    // Hide emissive rectangles from primary (camera) rays so the sky shows.
    if (depth == max_depth) {
        if (dynamic_cast<DiffuseLight*>(rec.mat.get()) != nullptr) {
            return g_sky.eval(normalize(r.direction));
        }
    }

    // Micro-normal perturb just for sand
    if (rec.mat == sand_ptr) {
        Vec3 g = g_bump.grad(rec.point * BUMP_FREQ);
        g = g - rec.normal * dot(g, rec.normal);
        rec.normal = normalize(rec.normal + BUMP_SCALE * g);
    }

    Vec3 emitted = rec.mat->emitted(rec);

    Ray scattered; Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered)) return emitted;

    if (attenuation.x < 1e-3 && attenuation.y < 1e-3 && attenuation.z < 1e-3) return emitted;

    if (depth < max_depth - 4) {
        double p = std::max({attenuation.x, attenuation.y, attenuation.z});
        p = std::max(0.05, clamp01(p));
        if (random_double() > p) return emitted;
        attenuation /= p;
    }

    Vec3 indirect = attenuation * ray_color(scattered, world, area_light, depth - 1, max_depth);

    Vec3 direct(0,0,0);
    Vec3 albedo;
    const int LIGHT_SAMPLES_PER_HIT = PREVIEW ? LIGHT_SAMPLES_PREVIEW : LIGHT_SAMPLES_FINAL;

    if (area_light && get_lambert_albedo(rec.mat, albedo)) {
        Vec3 L_light(0,0,0);
        for (int s=0; s<LIGHT_SAMPLES_PER_HIT; ++s) {
            Vec3 lp = area_light->sample_point();
            Vec3 toL = lp - rec.point;
            double dist2 = dot(toL, toL); if (dist2 <= 1e-12) continue;
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

// ====================== Cube builder ======================
static std::shared_ptr<Hittable> make_unit_cube(const Vec3& c, double edge, const std::shared_ptr<Material>& m) {
    double h = edge * 0.5;
    double x0 = c.x - h, x1 = c.x + h;
    double y0 = c.y - h, y1 = c.y + h;
    double z0 = c.z - h, z1 = c.z + h;
    auto list = std::make_shared<HittableList>();
    list->add(std::make_shared<XZRect>(x0, x1, z0, z1, y1, m));
    list->add(std::make_shared<XZRect>(x0, x1, z0, z1, y0, m));
    list->add(std::make_shared<XYRect>(x0, x1, y0, y1, z1, m));
    list->add(std::make_shared<XYRect>(x0, x1, y0, y1, z0, m));
    list->add(std::make_shared<YZRect>(y0, y1, z0, z1, x0, m));
    list->add(std::make_shared<YZRect>(y0, y1, z0, z1, x1, m));
    return list;
}

// ====================== Main ======================
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
    double vfov_deg = 35.0;
    double focus_dist = (lookfrom - lookat).length();
    Camera cam(lookfrom, lookat, Vec3(0,1,0), vfov_deg, aspect, 0.0, focus_dist, 0.0, 1.0);

    // BBox projector basis (from bbox.hpp)
    CameraBasis camBasis = make_camera_basis(lookfrom, lookat, Vec3(0,1,0), vfov_deg, aspect);

    // Materials
    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    auto sand   = sand_ptr;
    auto red_m  = std::make_shared<Lambertian>(Vec3(0.90, 0.18, 0.14));
    auto blu_m  = std::make_shared<Lambertian>(Vec3(0.18, 0.55, 0.95));
    auto grn_m  = std::make_shared<Lambertian>(Vec3(0.22, 0.84, 0.25));
    auto wht_m  = std::make_shared<Lambertian>(Vec3(0.85, 0.85, 0.85));
    auto sun    = std::make_shared<DiffuseLight>(Vec3(1.0, 0.98, 0.92), 8000.0);

    // Terrain
    const double XMIN=-22, XMAX=22, ZMIN=-22, ZMAX=22;
    const int NX = 96, NZ = 96;
    HeightField hf(XMIN, XMAX, ZMIN, ZMAX, NX, NZ, /*amp=*/1.8, /*scale=*/0.14, /*seed=*/1337);
    hf.generate();

    HittableList objects;

    // Side "sun"
    auto rect_light = std::make_shared<YZRect>( 2.0, 14.0, -25.0, 25.0, -30.0, sun);
    objects.add(rect_light);

    // FIXED CUBES (RGBW) — embedded & tilted
    struct CubeSpec { Vec3 center; double edge; const char* label; };
    const double base_edge = 1.6;
    std::vector<CubeSpec> specs = {
        { Vec3(-6.0, 0.0, -6.0), base_edge, "red"   },
        { Vec3( 6.0, 0.0, -6.0), base_edge, "green" },
        { Vec3(-6.0, 0.0,  6.0), base_edge, "blue"  },
        { Vec3( 6.0, 0.0,  6.0), base_edge, "white" }
    };

    std::mt19937 rng(1338);
    std::uniform_real_distribution<double> tilt_deg(-12.0, 12.0);
    std::uniform_real_distribution<double> sink_ratio_rng(0.08, 0.22);

    std::vector<CubePose> cubes; cubes.reserve(specs.size());

    for (auto& s : specs) {
        double sink_ratio = sink_ratio_rng(rng);
        double y_ground = hf.height_at(s.center.x, s.center.z);
        double y = y_ground + (1.0 - sink_ratio) * (s.edge*0.5);
        Vec3 c(s.center.x, y, s.center.z);

        hf.carve_footprint(c.x, c.z, s.edge,
                           /*sink=*/sink_ratio * s.edge,
                           /*rim_amp=*/0.10 * s.edge);

        auto mat = (std::string(s.label)=="red")?red_m: (std::string(s.label)=="green")?grn_m:
                   (std::string(s.label)=="blue")?blu_m:wht_m;
        auto cube = make_unit_cube(Vec3(0,0,0), s.edge, mat);

        double roll  = tilt_deg(rng);  // Z
        double pitch = tilt_deg(rng);  // X
        std::shared_ptr<Hittable> tilted = std::make_shared<RotateZ>(cube, roll);
        tilted = std::make_shared<RotateX>(tilted, pitch);
        tilted = std::make_shared<Translate>(tilted, c);

        objects.add(tilted);
        cubes.push_back(CubePose{c, s.edge, roll, pitch, s.label});
    }

    // Tessellate deformed sand
    hf.to_triangles(objects, sand);

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

    // 2D bounding boxes (drawn in yellow) — from bbox.hpp
    std::vector<Box2D> boxes; boxes.reserve(cubes.size());
    for (auto& pose : cubes){
        Box2D b = cube_bbox_screen_rot(camBasis, pose, width, height);
        if (b.valid) boxes.push_back(b);
    }
    for (auto& b : boxes){
        draw_rect(fb, width, height, b.x0,b.y0,b.x1,b.y1);
    }

    // Write image
    std::ofstream file("../out/image.ppm", std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(fb.data()), fb.size());
    std::cout << "Wrote ../out/image.ppm\n";

    // Write annotations (CSV + JSON)
    {
        std::ofstream c("../out/bboxes.csv");
        c << "label,xmin,ymin,xmax,ymax,width,height\n";
        for (auto& b : boxes){
            c << b.label << "," << b.x0 << "," << b.y0 << "," << b.x1 << "," << b.y1
              << "," << width << "," << height << "\n";
        }
        std::cout << "Wrote ../out/bboxes.csv\n";
    }
    {
        std::ofstream j("../out/bboxes.json");
        j << std::fixed << std::setprecision(0);
        j << "{\n  \"image_width\": " << width << ",\n  \"image_height\": " << height << ",\n  \"boxes\": [\n";
        for (size_t i=0;i<boxes.size();++i){
            auto& b = boxes[i];
            j << "    {\"label\": \"" << b.label << "\", \"xmin\": " << b.x0
              << ", \"ymin\": " << b.y0 << ", \"xmax\": " << b.x1 << ", \"ymax\": " << b.y1 << "}";
            if (i+1<boxes.size()) j << ",";
            j << "\n";
        }
        j << "  ]\n}\n";
        std::cout << "Wrote ../out/bboxes.json\n";
    }
    return 0;
}
