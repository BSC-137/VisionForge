// VisionForge — CLI wrapper: procedural desert + labeled cubes
// Outputs: <out>/image.ppm, inst.pgm, labels_from_mask.{csv,json}, bboxes.{csv,json}, labels_{yolo,coco}.*
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>

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

// dataset/annotation plumbing you already have
#include "visionforge/tag.hpp"
#include "visionforge/passes.hpp"
#include "visionforge/mask_writer.hpp"
#include "visionforge/bbox_from_mask.hpp"
#include "visionforge/coco.hpp"
#include "visionforge/yolo.hpp"


#include "visionforge/triangle.hpp"
#include "visionforge/transform.hpp"


#ifndef PI
#define PI 3.14159265358979323846
#endif

// ---------------------------- Small helpers ----------------------------
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

static bool parse_vec3_csv(const std::string& s, Vec3& out) {
    double x=0,y=0,z=0; char c1=0,c2=0; std::stringstream ss(s);
    if ((ss >> x >> c1 >> y >> c2 >> z) && c1==',' && c2==',') { out=Vec3(x,y,z); return true; }
    return false;
}
static bool parse_double_csv2(const std::string& s, double& a, double& b){
    char c; std::stringstream ss(s); return (ss>>a>>c>>b) && (c==',');
}
static bool parse_double_csv4(const std::string& s, double& a,double& b,double& c,double& d){
    char c1,c2,c3; std::stringstream ss(s); return (ss>>a>>c1>>b>>c2>>c>>c3>>d) && (c1==',' && c2==',' && c3==',');
}
static bool parse_bool(const std::string& s){
    std::string t=s; std::transform(t.begin(),t.end(),t.begin(),::tolower);
    return (t=="1"||t=="true"||t=="yes"||t=="on");
}

// Color parsing: names, #RRGGBB, "r,g,b" (0..1 or 0..255 auto-detect)
static Vec3 parse_color_one(const std::string& token, bool& ok, std::string& labelOut){
    static const std::map<std::string, Vec3> named = {
        {"red",   Vec3(0.90,0.18,0.14)}, {"green", Vec3(0.22,0.84,0.25)},
        {"blue",  Vec3(0.18,0.55,0.95)}, {"white", Vec3(0.85,0.85,0.85)},
        {"sand",  Vec3(0.78,0.72,0.58)}, {"yellow",Vec3(0.95,0.90,0.12)},
        {"magenta",Vec3(0.93,0.18,0.90)},{"cyan",  Vec3(0.10,0.85,0.90)},
        {"gray",  Vec3(0.5,0.5,0.5)}
    };
    ok=true; labelOut=token;
    std::string t=token; std::transform(t.begin(),t.end(),t.begin(),::tolower);

    if (t.size()==7 && t[0]=='#') {
        auto hex2 = [&](char a, char b)->int{
            auto v=[&](char c)->int{
                if (c>='0'&&c<='9') return c-'0';
                if (c>='a'&&c<='f') return c-'a'+10;
                if (c>='A'&&c<='F') return c-'A'+10;
                return 0;
            };
            return v(a)*16+v(b);
        };
        int r=hex2(t[1],t[2]), g=hex2(t[3],t[4]), b=hex2(t[5],t[6]);
        return Vec3(r/255.0, g/255.0, b/255.0);
    }

    auto it = named.find(t);
    if (it != named.end()) return it->second;

    // "r,g,b"
    {
        double r=0,g=0,b=0; char c1=0,c2=0; std::stringstream ss(token);
        if ((ss>>r>>c1>>g>>c2>>b) && c1==',' && c2==',') {
            // if any component > 1 assume 0..255
            if (r>1.0 || g>1.0 || b>1.0) { r/=255.0; g/=255.0; b/=255.0; }
            return Vec3(clamp01(r),clamp01(g),clamp01(b));
        }
    }

    ok=false; labelOut="cube";
    return Vec3(0.7,0.7,0.7);
}
static std::vector<std::string> split_color_list(const std::string& s){
    std::vector<std::string> out;
    // If ';' exists, use it as the top-level separator; else fall back to commas.
    char sep = (s.find(';') != std::string::npos) ? ';' : ',';
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}


// ---------------------------- CLI Options ----------------------------
struct Opts {
    // IO
    std::string out = "out";
    std::string name = "run1";

    // Image & sampling
    int width = 1280, height = 720;
    int spp = 96, max_depth = 16;
    bool preview = false;
    unsigned seed = 1337;

    // Exposure & sky
    double exposure_comp = 6.5;
    double sky_gain = 45.0;
    double sun_az_deg = 300.0;
    double sun_el_deg = 12.0;
    double turbidity = 3.5;

    // Camera
    Vec3 lookfrom = Vec3(18.0,8.0,24.0);
    Vec3 lookat   = Vec3(0.0,1.2,0.0);
    double fov_deg = 35.0;

    // Terrain
    double terrain_amp = 1.8;
    double terrain_scale = 0.14;
    int terrain_nx = 96, terrain_nz = 96;

    // Sand micro bump
    double sand_bump_freq = 5.0;
    double sand_bump_scale = 0.22;

    // World bounds for random placement
    double XMIN=-22, XMAX=22, ZMIN=-22, ZMAX=22;

    // Cubes
    int cubes = 4;
    double cube_edge_min = 1.4, cube_edge_max = 1.9;
    double cube_tilt_abs = 12.0;
    std::string cube_colors_csv = "red,green,blue,white";
    std::string cube_placement = "grid"; // or "random"
    double cube_min_spacing = 3.5;       // for random placement

    // Light (rect on YZ plane at fixed x)
    double light_x = -30.0;
    double light_y0 = 2.0, light_y1 = 14.0;
    double light_z0 = -25.0, light_z1 = 25.0;
    Vec3   light_color = Vec3(1.0, 0.98, 0.92);
    double light_intensity = 8000.0;
    bool   show_light_on_primary = false;   // if false: hide emissive rect from camera hit
    bool   match_sky_to_light = true;

    // Adaptive sampling
    int    min_spp = 6;
    double rel_noise_target = 0.020;

    // Light sampling
    int light_samples_preview = 2;
    int light_samples_final   = 6;
    int spp_preview = 12;

    // Dataset toggles (keeping default on)
    bool write_bbox_overlay = true;
};

static void print_usage() {
    std::cout <<
R"(VisionForge — procedural desert demo

Usage:
  visionforge [--out DIR] [--width W] [--height H] [--spp N]
              [--max-depth N] [--seed S] [--preview]
              [--exposure X] [--sky-gain X]
              [--sun-az DEG] [--sun-el DEG] [--turbidity T]
              [--lookfrom "x,y,z"] [--lookat "x,y,z"] [--fov DEG]
              [--terrain-amp A] [--terrain-scale S] [--terrain-nx N] [--terrain-nz N]
              [--sand-bump-freq F] [--sand-bump-scale S]
              [--world-bounds "xmin,xmax,zmin,zmax"]
              [--cubes N] [--cube-colors "name|#hex|r,g,b,..."]
              [--cube-edge-min E] [--cube-edge-max E]
              [--tilt-abs DEG] [--placement grid|random] [--min-spacing D]
              [--light-x X] [--light-y "y0,y1"] [--light-z "z0,z1"]
              [--light-color "r,g,b|#hex|name"] [--light-intensity I]
              [--show-light true|false] [--match-sky true|false]
              [--help]

Examples:
  visionforge --cubes 9 --placement random --cube-colors "red,#3cb371,0.1,0.2,0.9"
  visionforge --terrain-amp 2.2 --sand-bump-scale 0.35 --tilt-abs 18
)";
}

static Opts parse(int argc, char** argv) {
    Opts o;
    auto need = [&](int &i){ if(i+1>=argc){ print_usage(); std::exit(2);} return ++i; };

    for (int i=1;i<argc;++i) {
        std::string a(argv[i]);
        if (a=="--out")               o.out = argv[need(i)];
        else if (a=="--width")        o.width = std::stoi(argv[need(i)]);
        else if (a=="--height")       o.height = std::stoi(argv[need(i)]);
        else if (a=="--spp")          o.spp = std::stoi(argv[need(i)]);
        else if (a=="--max-depth")    o.max_depth = std::stoi(argv[need(i)]);
        else if (a=="--seed")         o.seed = static_cast<unsigned>(std::stoul(argv[need(i)]));
        else if (a=="--preview")      o.preview = true;
        else if (a=="--exposure")     o.exposure_comp = std::stod(argv[need(i)]);
        else if (a=="--sky-gain")     o.sky_gain = std::stod(argv[need(i)]);
        else if (a=="--sun-az")       o.sun_az_deg = std::stod(argv[need(i)]);
        else if (a=="--sun-el")       o.sun_el_deg = std::stod(argv[need(i)]);
        else if (a=="--turbidity")    o.turbidity = std::stod(argv[need(i)]);
        else if (a=="--lookfrom")    { Vec3 v; if(!parse_vec3_csv(argv[need(i)],v)){ std::cerr<<"Bad --lookfrom\n"; std::exit(2);} o.lookfrom=v; }
        else if (a=="--lookat")      { Vec3 v; if(!parse_vec3_csv(argv[need(i)],v)){ std::cerr<<"Bad --lookat\n";   std::exit(2);} o.lookat=v; }
        else if (a=="--fov")          o.fov_deg = std::stod(argv[need(i)]);

        else if (a=="--terrain-amp")    o.terrain_amp = std::stod(argv[need(i)]);
        else if (a=="--terrain-scale")  o.terrain_scale = std::stod(argv[need(i)]);
        else if (a=="--terrain-nx")     o.terrain_nx = std::stoi(argv[need(i)]);
        else if (a=="--terrain-nz")     o.terrain_nz = std::stoi(argv[need(i)]);
        else if (a=="--sand-bump-freq") o.sand_bump_freq = std::stod(argv[need(i)]);
        else if (a=="--sand-bump-scale")o.sand_bump_scale= std::stod(argv[need(i)]);
        else if (a=="--world-bounds") {
            double a,b,c,d; if(!parse_double_csv4(argv[need(i)],a,b,c,d)){ std::cerr<<"Bad --world-bounds\n"; std::exit(2); }
            o.XMIN=a; o.XMAX=b; o.ZMIN=c; o.ZMAX=d;
        }

        else if (a=="--cubes")           o.cubes = std::stoi(argv[need(i)]);
        else if (a=="--cube-colors")     o.cube_colors_csv = argv[need(i)];
        else if (a=="--cube-edge-min")   o.cube_edge_min = std::stod(argv[need(i)]);
        else if (a=="--cube-edge-max")   o.cube_edge_max = std::stod(argv[need(i)]);
        else if (a=="--tilt-abs")        o.cube_tilt_abs = std::stod(argv[need(i)]);
        else if (a=="--placement")       o.cube_placement = argv[need(i)];
        else if (a=="--min-spacing")     o.cube_min_spacing = std::stod(argv[need(i)]);

        else if (a=="--light-x")         o.light_x = std::stod(argv[need(i)]);
        else if (a=="--light-y")        { double y0,y1; if(!parse_double_csv2(argv[need(i)],y0,y1)){ std::cerr<<"Bad --light-y\n"; std::exit(2);} o.light_y0=y0; o.light_y1=y1; }
        else if (a=="--light-z")        { double z0,z1; if(!parse_double_csv2(argv[need(i)],z0,z1)){ std::cerr<<"Bad --light-z\n"; std::exit(2);} o.light_z0=z0; o.light_z1=z1; }
        else if (a=="--light-color")    {
            bool ok=true; std::string lbl;
            o.light_color = parse_color_one(argv[need(i)], ok, lbl);
            if(!ok){ std::cerr<<"Bad --light-color\n"; std::exit(2); }
        }
        else if (a=="--light-intensity") o.light_intensity = std::stod(argv[need(i)]);
        else if (a=="--show-light")      o.show_light_on_primary = parse_bool(argv[need(i)]);
        else if (a=="--match-sky")       o.match_sky_to_light = parse_bool(argv[need(i)]);

        else if (a=="--help" || a=="-h"){ print_usage(); std::exit(0); }
        else { std::cerr << "Unknown flag: " << a << "\n"; print_usage(); std::exit(2); }
    }
    std::filesystem::create_directories(o.out);
    return o;
}

// ---------------------------- Procedural terrain ----------------------------
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
                s += a * n; nrm += a; a *= 0.62; f *= 1.9;
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
    void carve_footprint(double x0, double z0, double edge, double sink, double rim_amp){
        double r = 0.6 * edge; double r2 = r*r; double r_out = r*1.8;
        for (int k=0;k<=nz;++k){
            double z = zmin + (zmax-zmin)*(double(k)/nz);
            for (int i=0;i<=nx;++i){
                double x = xmin + (xmax-xmin)*(double(i)/nx);
                double dx=x-x0, dz=z-z0; double d2=dx*dx+dz*dz; if (d2 > r_out*r_out) continue;
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

// ---------------------------- Globals that need CLI control ----------------------------
static vf::perlin g_bump(424242);
static Sky g_sky(/*az*/300.0, /*el*/12.0, /*turb*/3.5);
static std::shared_ptr<Lambertian> sand_ptr;

// Material helper
static inline bool get_lambert_albedo(const std::shared_ptr<Material>& m, Vec3& out_albedo) {
    if (auto* lam = dynamic_cast<Lambertian*>(m.get())) { out_albedo = lam->albedo; return true; }
    return false;
}

// --------------- Integrator (unchanged except toggles wired to CLI) ---------------
template<typename RectT>
Vec3 ray_color(const Ray& r, const Hittable& world, const RectT* area_light, int depth, int max_depth,
               bool PREVIEW, int LIGHT_SAMPLES_PREVIEW, int LIGHT_SAMPLES_FINAL,
               int MAX_DEPTH_PREVIEW, int MAX_DEPTH_FINAL, double SKY_VIEW_GAIN,
               bool show_light_on_primary,
               GBuffer* gbuf, int pix_index)
{
    if (depth <= 0) return Vec3(0,0,0);
    HitRecord rec;

    // sky on miss (primary only)
    if (!world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        if (depth == max_depth) return g_sky.eval(normalize(r.direction)) * SKY_VIEW_GAIN;
        return Vec3(0,0,0);
    }

    // write instance id for primary
    if (depth == max_depth && gbuf) {
        uint32_t inst = (rec.hit_object ? rec.hit_object->obj.instance_id : 0);
        gbuf->inst_id[pix_index] = inst;
    }

    // optionally hide emissive rect from primary (so we see sky)
    if (!show_light_on_primary && depth == max_depth) {
        if (dynamic_cast<DiffuseLight*>(rec.mat.get()) != nullptr)
            return g_sky.eval(normalize(r.direction)) * SKY_VIEW_GAIN;
    }

    // micro bump only for sand material; frequency/scale set later via globals
    if (rec.mat == sand_ptr) {
        Vec3 g = g_bump.grad(rec.point * /*freq set later*/ 1.0); // scaled at call-site
        // note: we fold freq and scale below
    }

    Vec3 emitted = rec.mat->emitted(rec);

    Ray scattered; Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered)) return emitted;

    if (attenuation.x < 1e-3 && attenuation.y < 1e-3 && attenuation.z < 1e-3) return emitted;

    // RR
    if (depth < max_depth - 4) {
        double p = std::max({attenuation.x, attenuation.y, attenuation.z});
        p = std::max(0.05, std::min(1.0, p));
        if (random_double() > p) return emitted;
        attenuation /= p;
    }

    Vec3 indirect = attenuation * ray_color(scattered, world, area_light, depth - 1, max_depth,
                                            PREVIEW, LIGHT_SAMPLES_PREVIEW, LIGHT_SAMPLES_FINAL,
                                            MAX_DEPTH_PREVIEW, MAX_DEPTH_FINAL, SKY_VIEW_GAIN,
                                            show_light_on_primary, gbuf, pix_index);

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

    // Apply sand micro-bumps here (we delayed to know globals outside)
    return emitted + direct + indirect;
}

// Micro-bump globals (wired by CLI)
static double G_BUMP_FREQ = 5.0;
static double G_BUMP_SCALE = 0.22;

// Slightly modified Lambertian normal perturbation for sand
static void apply_sand_bump(HitRecord& rec) {
    if (rec.mat != sand_ptr) return;
    Vec3 g = g_bump.grad(rec.point * G_BUMP_FREQ);
    g = g - rec.normal * dot(g, rec.normal);
    rec.normal = normalize(rec.normal + G_BUMP_SCALE * g);
}

// ---------------------------- Cube builders ----------------------------
static std::shared_ptr<Hittable> make_unit_cube(const Vec3& c, double edge, const std::shared_ptr<Material>& m) {
    double h = edge * 0.5;
    double x0 = c.x - h, x1 = c.x + h;
    double y0 = c.y - h, y1 = c.y + h;
    double z0 = c.z - h, z1 = c.z + h;
    auto list = std::make_shared<HittableList>();
    list->add(std::make_shared<XZRect>(x0, x1, z0, z1, y1, m)); // top
    list->add(std::make_shared<XZRect>(x0, x1, z0, z1, y0, m)); // bottom
    list->add(std::make_shared<XYRect>(x0, x1, y0, y1, z1, m)); // front
    list->add(std::make_shared<XYRect>(x0, x1, y0, y1, z0, m)); // back
    list->add(std::make_shared<YZRect>(y0, y1, z0, z1, x0, m)); // left
    list->add(std::make_shared<YZRect>(y0, y1, z0, z1, x1, m)); // right
    return list;
}

// Manifest
static void write_manifest(const std::string& outdir, const Opts& o, int actual_avg_spp, double seconds) {
    std::ofstream j(outdir + "/manifest.json");
    j << std::fixed << std::setprecision(3);
    j << "{\n"
      << "  \"tool\": \"VisionForge\",\n"
      << "  \"width\": " << o.width << ",\n"
      << "  \"height\": " << o.height << ",\n"
      << "  \"spp_target\": " << o.spp << ",\n"
      << "  \"spp_avg\": " << actual_avg_spp << ",\n"
      << "  \"max_depth\": " << o.max_depth << ",\n"
      << "  \"seed\": " << o.seed << ",\n"
      << "  \"seconds\": " << seconds << "\n"
      << "}\n";
}

// ---------------------------- MAIN ----------------------------
int main(int argc, char** argv) {
    Opts o = parse(argc, argv);
    std::srand(o.seed);

    // render controls
    bool   PREVIEW = o.preview;
    int    WIDTH   = o.width;
    int    HEIGHT  = o.height;
    int    SPP_PREVIEW = o.spp_preview;
    int    SPP_FINAL   = o.spp;
    int    MAX_DEPTH_PREVIEW = 6;
    int    MAX_DEPTH_FINAL   = o.max_depth;
    int    LIGHT_SAMPLES_PREVIEW = o.light_samples_preview;
    int    LIGHT_SAMPLES_FINAL   = o.light_samples_final;
    int    MIN_SPP = o.min_spp;
    double REL_NOISE_TARGET = o.rel_noise_target;

    // exposure / sky intensity
    double F_NUMBER = 2.0;
    double SHUTTER  = 1.0/30.0;
    int    ISO      = 400;
    double EXPOSURE_COMP = o.exposure_comp;
    double SKY_VIEW_GAIN = o.sky_gain;

    // image dims
    const int width  = WIDTH;
    const int height = HEIGHT;
    const int spp_target = PREVIEW ? SPP_PREVIEW : SPP_FINAL;
    const int max_depth  = PREVIEW ? MAX_DEPTH_PREVIEW : MAX_DEPTH_FINAL;
    const double aspect  = double(width)/double(height);

    // camera
    Vec3 lookfrom = o.lookfrom;
    Vec3 lookat   = o.lookat;
    double vfov_deg = o.fov_deg;
    double focus_dist = (lookfrom - lookat).length();
    Camera cam(lookfrom, lookat, Vec3(0,1,0), vfov_deg, aspect, 0.0, focus_dist, 0.0, 1.0);
    CameraBasis camBasis = make_camera_basis(lookfrom, lookat, Vec3(0,1,0), vfov_deg, aspect);

    // sky (CLI)
    g_sky.set_angles(o.sun_az_deg, o.sun_el_deg); // requires small setter; see notes below
    g_sky.set_turbidity(o.turbidity);

    // materials
    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    auto red_m  = std::make_shared<Lambertian>(Vec3(0.90, 0.18, 0.14));
    auto blu_m  = std::make_shared<Lambertian>(Vec3(0.18, 0.55, 0.95));
    auto grn_m  = std::make_shared<Lambertian>(Vec3(0.22, 0.84, 0.25));
    auto wht_m  = std::make_shared<Lambertian>(Vec3(0.85, 0.85, 0.85));

    // sand bump controls
    G_BUMP_FREQ  = o.sand_bump_freq;
    G_BUMP_SCALE = o.sand_bump_scale;

    // terrain
    HeightField hf(o.XMIN, o.XMAX, o.ZMIN, o.ZMAX, o.terrain_nx, o.terrain_nz, o.terrain_amp, o.terrain_scale, /*seed*/1337);
    hf.generate();

    HittableList objects;

    // area light (YZRect at x = o.light_x)
    auto sun_em = std::make_shared<DiffuseLight>(o.light_color, o.light_intensity);
    auto rect_light = std::make_shared<YZRect>(
        /*y0=*/o.light_y0, /*y1=*/o.light_y1,
        /*z0=*/o.light_z0, /*z1=*/o.light_z1,
        /*x=*/ o.light_x,
        sun_em
    );
    objects.add(rect_light);

    if (o.match_sky_to_light) {
        g_sky.set_sun_from_dir(rect_light->light_normal()); // keep your original sync
    }

    // parse cube colors
    std::vector<Vec3> cube_colors;
    std::vector<std::string> color_labels;
    {
        // NOTE: split_color_list lets you separate tokens with ';'
        // so "0.1,0.2,0.9" stays one token if you ever use raw RGB.
        auto toks = split_color_list(o.cube_colors_csv);
        if (toks.empty()) toks = {"red","green","blue","white"};

        cube_colors.reserve(toks.size());
        color_labels.reserve(toks.size());

        for (const auto& t : toks){
            bool ok = true;
            std::string lbl;
            Vec3 c = parse_color_one(t, ok, lbl);
            if (!ok) {
                std::cerr << "Warning: bad color token '" << t << "', using gray.\n";
                c = Vec3(0.5,0.5,0.5);
                lbl = "cube";
            }
            cube_colors.push_back(c);
            color_labels.push_back(lbl);
        }
    }


    // Build lambertian materials for each unique color token
    std::vector<std::shared_ptr<Lambertian>> cube_mats;
    cube_mats.reserve(cube_colors.size());
    for (auto& c : cube_colors) cube_mats.push_back(std::make_shared<Lambertian>(c));

    // cube placement
    struct CubePose { Vec3 c; double edge; double roll; double pitch; int color_idx; std::string label; };
    std::vector<CubePose> cubes; cubes.reserve(o.cubes);
    std::mt19937 rng(o.seed ^ 0x9e3779b1);
    std::uniform_real_distribution<double> edge_rng(o.cube_edge_min, o.cube_edge_max);
    std::uniform_real_distribution<double> tilt_rng(-o.cube_tilt_abs, o.cube_tilt_abs);
    std::uniform_real_distribution<double> sink_ratio_rng(0.08, 0.22);

    auto place_grid = [&](){
        int n = (int)std::ceil(std::sqrt((double)o.cubes));
        double dx = (o.XMAX - o.XMIN) / (n+1);
        double dz = (o.ZMAX - o.ZMIN) / (n+1);
        int count=0;
        for (int gz=0; gz<n && count<o.cubes; ++gz) {
            for (int gx=0; gx<n && count<o.cubes; ++gx) {
                double x = o.XMIN + (gx+1)*dx;
                double z = o.ZMIN + (gz+1)*dz;
                double e = edge_rng(rng);
                double y_ground = hf.height_at(x,z);
                double sink_ratio = sink_ratio_rng(rng);
                double y = y_ground + (1.0 - sink_ratio) * (e*0.5);
                hf.carve_footprint(x, z, e, sink_ratio * e, 0.10 * e);
                int ci = count % (int)cube_mats.size();
                cubes.push_back({Vec3(x,y,z), e, tilt_rng(rng), tilt_rng(rng), ci, color_labels[ci]});
                ++count;
            }
        }
    };
    auto place_random = [&](){
        std::uniform_real_distribution<double> xr(o.XMIN+2.0, o.XMAX-2.0);
        std::uniform_real_distribution<double> zr(o.ZMIN+2.0, o.ZMAX-2.0);
        auto far_enough = [&](double x,double z){
            for (auto& p: cubes){
                double dx=x-p.c.x, dz=z-p.c.z;
                if (std::sqrt(dx*dx+dz*dz) < o.cube_min_spacing) return false;
            }
            return true;
        };
        int tries=0;
        while ((int)cubes.size() < o.cubes && tries < 10000){
            double x=xr(rng), z=zr(rng);
            if (!far_enough(x,z)){ ++tries; continue; }
            double e = edge_rng(rng);
            double y_ground = hf.height_at(x,z);
            double sink_ratio = sink_ratio_rng(rng);
            double y = y_ground + (1.0 - sink_ratio) * (e*0.5);
            hf.carve_footprint(x, z, e, sink_ratio * e, 0.10 * e);
            int ci = (int)cubes.size() % (int)cube_mats.size();
            cubes.push_back({Vec3(x,y,z), e, tilt_rng(rng), tilt_rng(rng), ci, color_labels[ci]});
        }
        if ((int)cubes.size() < o.cubes) {
            std::cerr<<"Warning: random placement hit try limit; placed "<<cubes.size()<<"/"<<o.cubes<<"\n";
        }
    };

    std::string placement = o.cube_placement; std::transform(placement.begin(),placement.end(),placement.begin(),::tolower);
    if (placement=="random") place_random(); else place_grid();

    // add cubes (tag instance ids)
    IdRegistry reg;
    uint32_t next_instance_id = 1; // 0=background
    for (size_t idx=0; idx<cubes.size(); ++idx) {
        auto& cp = cubes[idx];
        auto mat = cube_mats[cp.color_idx];

        auto cube = make_unit_cube(Vec3(0,0,0), cp.edge, mat);
        std::shared_ptr<Hittable> tilted = std::make_shared<RotateZ>(cube, cp.roll);
        tilted = std::make_shared<RotateX>(tilted, cp.pitch);
        tilted = std::make_shared<Translate>(tilted, cp.c);

        uint32_t id = next_instance_id++;
        std::string label = cp.label.empty()? "cube" : cp.label;
        ObjectInfo info{ id, /*class_id=*/1u, label.c_str() };
        auto tagged = std::make_shared<Tag>(tilted, info);
        objects.add(tagged);

        reg.id_to_label[id] = label;
        reg.id_to_class[id] = 1u;
    }

    // tessellate sand
    hf.to_triangles(objects, sand_ptr);

    // BVH
    std::vector<std::shared_ptr<Hittable>> objs_vec = objects.objects;
    BVHNode world(objs_vec, 0, objs_vec.size());

    // render
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> fb(width * height * 3, 0);
    GBuffer gbuf(width, height);
    double exposure = exposure_scale(F_NUMBER, SHUTTER, ISO) * EXPOSURE_COMP;

    long long spp_accum = 0;

    #if defined(VF_USE_OMP)
      #pragma omp parallel for schedule(dynamic, 1) reduction(+:spp_accum)
    #endif
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            Vec3 sum(0,0,0);
            double meanL = 0.0, M2 = 0.0;
            int s = 0;
            int pix_index = ((height-1-j)*width + i);
            for (; s < spp_target; ++s) {
                double u = (i + random_double()) / (width  - 1);
                double v = (j + random_double()) / (height - 1);
                Ray r = cam.get_ray(u, v);
                Vec3 c = ray_color(r, world, rect_light.get(), max_depth, max_depth,
                                   PREVIEW, LIGHT_SAMPLES_PREVIEW, LIGHT_SAMPLES_FINAL,
                                   MAX_DEPTH_PREVIEW, MAX_DEPTH_FINAL, SKY_VIEW_GAIN,
                                   o.show_light_on_primary, &gbuf, pix_index);
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
            spp_accum += (s+1);

            Vec3 pixel = (sum / double(s+1)) * exposure;
            Vec3 mapped = aces_tonemap(pixel);
            mapped = Vec3(std::sqrt(mapped.x), std::sqrt(mapped.y), std::sqrt(mapped.z));
            int idx = pix_index*3;
            fb[idx+0] = (unsigned char)(256 * clamp01(mapped.x));
            fb[idx+1] = (unsigned char)(256 * clamp01(mapped.y));
            fb[idx+2] = (unsigned char)(256 * clamp01(mapped.z));
        }
    }

    // 2D bbox overlay from analytic cube OBBs
    std::vector<Box2D> boxes; boxes.reserve(cubes.size());
    for (auto& pose : cubes){
        Box2D b = cube_bbox_screen_rot(camBasis, /*pose*/{pose.c, pose.edge, pose.roll, pose.pitch, pose.label.c_str()}, width, height);
        if (b.valid) boxes.push_back(b);
    }
    if (o.write_bbox_overlay){
        for (auto& b : boxes) draw_rect(fb, width, height, b.x0,b.y0,b.x1,b.y1);
    }

    // write outputs
    const std::string ppm  = o.out + "/image.ppm";
    const std::string csv  = o.out + "/bboxes.csv";
    const std::string json = o.out + "/bboxes.json";

    std::ofstream file(ppm, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(fb.data()), fb.size());
    std::cout << "Wrote " << ppm << "\n";

    {
        std::ofstream c(csv);
        c << "label,xmin,ymin,xmax,ymax,width,height\n";
        for (auto& b : boxes){
            c << b.label << "," << b.x0 << "," << b.y0 << "," << b.x1 << "," << b.y1
              << "," << width << "," << height << "\n";
        }
        std::cout << "Wrote " << csv << "\n";
    }
    {
        std::ofstream j(json);
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
        std::cout << "Wrote " << json << "\n";
    }

    // Instance mask + tight boxes from mask
    write_inst_pgm(o.out + "/inst.pgm", gbuf);
    std::cout << "Wrote " << (o.out + "/inst.pgm") << "\n";

    auto tight = boxes_from_mask(gbuf, reg);
    write_boxes_csv (o.out + "/labels_from_mask.csv",  tight, width, height);
    write_boxes_json(o.out + "/labels_from_mask.json", tight, width, height);
    std::cout << "Wrote labels_from_mask.{csv,json}\n";

    // YOLO / COCO
    {
        std::vector<std::tuple<int,int,int,int,int>> yolo_boxes;
        for (auto& b : tight) {
            yolo_boxes.emplace_back(int(b.class_id), b.x0, b.y0, b.x1, b.y1);
        }
        write_yolo_txt(o.out + "/labels_yolo.txt", width, height, yolo_boxes);
        std::cout << "Wrote " << (o.out + "/labels_yolo.txt") << "\n";
    }
    {
        CocoWriter coco;
        coco.ensure_category(1, "cube");
        int img_id = coco.add_image("image.ppm", width, height);
        for (auto& b : tight) {
            coco.add_box(img_id, b.class_id, b.x0, b.y0, b.x1, b.y1, b.instance_id, b.label);
        }
        coco.write(o.out + "/labels_coco.json");
        std::cout << "Wrote " << (o.out + "/labels_coco.json") << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    int avg_spp = int(double(spp_accum) / double(width*height) + 0.5);
    write_manifest(o.out, o, avg_spp, secs);
    std::cout << "Done. avg_spp=" << avg_spp << ", time=" << secs << "s\n";
    return 0;
}
