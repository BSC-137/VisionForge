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
#include "visionforge/tag.hpp"
#include "visionforge/passes.hpp"
#include "visionforge/mask_writer.hpp"
#include "visionforge/bbox_from_mask.hpp"
#include "visionforge/coco.hpp"
#include "visionforge/yolo.hpp"
#include "visionforge/triangle.hpp"
#include "visionforge/transform.hpp"
#include "visionforge/mesh.hpp"
#include "visionforge/exr_writer.hpp"
#include "visionforge/png_writer.hpp"
#include "visionforge/world_config.hpp"
#include "visionforge/scene_graph.hpp"
#include "visionforge/pbr_material.hpp"
#include "visionforge/terrain.hpp"
#include "visionforge/placement.hpp"
#include "visionforge/hdr_sky.hpp"
#include "visionforge/image_texture.hpp"
#include "visionforge/triplanar_material.hpp"
#include "visionforge/asset_manager.hpp"
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>


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
    int width = 320, height = 180;
    int spp = 1, max_depth = 6;
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

    // Mesh loading
    std::string obj_path;
    Vec3 obj_pos = Vec3(0, 0, 0);
    double obj_scale = 1.0;
    std::string obj_color = "white";
    double obj_sink = 0.05;

    // EXR / depth export
    bool write_exr  = false;
    bool depth_only = false;

    // Dataset toggles (keeping default on)
    bool write_bbox_overlay = true;

    // HDR sky
    std::string hdr_path;
    double hdr_intensity = 1.0;

    // Ground texture (triplanar)
    std::string ground_tex_path;
    double ground_scale = 2.0;
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
              [--obj PATH] [--obj-pos "x,y,z"] [--obj-scale S]
              [--obj-color "name|#hex|r,g,b"] [--sink D]
              [--hdr PATH] [--hdr-intensity F]
              [--ground-tex PATH] [--ground-scale F]
              [--exr] [--depth-only]
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

        else if (a=="--obj")             o.obj_path = argv[need(i)];
        else if (a=="--obj-pos")        { Vec3 v; if(!parse_vec3_csv(argv[need(i)],v)){ std::cerr<<"Bad --obj-pos\n"; std::exit(2);} o.obj_pos=v; }
        else if (a=="--obj-scale")       o.obj_scale = std::stod(argv[need(i)]);
        else if (a=="--obj-color")       o.obj_color = argv[need(i)];
        else if (a=="--sink")            o.obj_sink = std::stod(argv[need(i)]);

        else if (a=="--hdr")             o.hdr_path = argv[need(i)];
        else if (a=="--hdr-intensity")   o.hdr_intensity = std::stod(argv[need(i)]);
        else if (a=="--ground-tex")      o.ground_tex_path = argv[need(i)];
        else if (a=="--ground-scale")    o.ground_scale = std::stod(argv[need(i)]);

        else if (a=="--exr")             o.write_exr = true;
        else if (a=="--depth-only")    { o.depth_only = true; o.write_exr = true; }

        else if (a=="--help" || a=="-h"){ print_usage(); std::exit(0); }
        else { std::cerr << "Unknown flag: " << a << "\n"; print_usage(); std::exit(2); }
    }
    std::filesystem::create_directories(o.out);
    return o;
}

// HeightField is now in include/visionforge/terrain.hpp
// SlopeAlign + snap_y are in include/visionforge/placement.hpp

// ---------------------------- Globals that need CLI control ----------------------------
static vf::perlin g_bump(424242);
static Sky g_sky(/*az*/300.0, /*el*/12.0, /*turb*/3.5);
static std::shared_ptr<Lambertian> sand_ptr;
static std::shared_ptr<HDRSky> g_hdr_sky;
static std::shared_ptr<Material> g_terrain_mat;

static inline double pbr_saturate(double x) { return std::clamp(x, 0.0, 1.0); }

static inline Vec3 fresnel_schlick(double cos_theta, const Vec3& F0) {
    const double x = pbr_saturate(1.0 - cos_theta);
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double t = x4 * x;
    return F0 + (Vec3(1.0, 1.0, 1.0) - F0) * t;
}

static inline double ggx_distribution(double NdotH, double roughness) {
    const double a = std::max(0.02, roughness * roughness);
    const double a2 = a * a;
    const double d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / std::max(PI * d * d, 1e-8);
}

static inline double schlick_beckmann_g1(double NdotX, double roughness) {
    const double r = roughness + 1.0;
    const double k = (r * r) / 8.0;
    return NdotX / std::max(NdotX * (1.0 - k) + k, 1e-6);
}

static inline double schlick_beckmann_geometry(double NdotV, double NdotL, double roughness) {
    return schlick_beckmann_g1(NdotV, roughness) * schlick_beckmann_g1(NdotL, roughness);
}

static inline Vec3 cook_torrance_brdf(const Material& m, const Vec3& N, const Vec3& V, const Vec3& L) {
    const Vec3 VH = V + L;
    if (VH.length_squared() < 1e-12) return Vec3(0, 0, 0);
    const Vec3 H = normalize(VH);
    const double NdotL = pbr_saturate(dot(N, L));
    const double NdotV = pbr_saturate(dot(N, V));
    const double NdotH = pbr_saturate(dot(N, H));
    const double VdotH = pbr_saturate(dot(V, H));

    const double roughness = std::clamp(m.roughness, 0.0, 1.0);
    const double metallic = std::clamp(m.metallic, 0.0, 1.0);
    const Vec3 base = m.base_color;

    const Vec3 dielectric_F0(0.04, 0.04, 0.04);
    const Vec3 F0 = dielectric_F0 * (1.0 - metallic) + base * metallic;
    const Vec3 F = fresnel_schlick(VdotH, F0);
    const double D = ggx_distribution(NdotH, roughness);
    const double G = schlick_beckmann_geometry(NdotV, NdotL, roughness);

    const Vec3 numerator = (D * G) * F;
    const double denom = std::max(4.0 * NdotV * NdotL, 1e-6);
    const Vec3 specular = numerator / denom;

    const Vec3 kS = F;
    const Vec3 kD = (Vec3(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);
    const Vec3 diffuse = (kD * base) / PI;
    return diffuse + specular;
}

static inline double pbr_brdf_pdf(const Material& m, const Vec3& N, const Vec3& V, const Vec3& L) {
    const Vec3 VH = V + L;
    if (VH.length_squared() < 1e-12) return 0.0;
    const Vec3 H = normalize(VH);
    const double NdotL = pbr_saturate(dot(N, L));
    const double NdotH = pbr_saturate(dot(N, H));
    const double VdotH = pbr_saturate(dot(V, H));
    if (NdotL <= 0.0 || NdotH <= 0.0 || VdotH <= 0.0) return 0.0;

    const double roughness = std::clamp(m.roughness, 0.0, 1.0);
    const double D = ggx_distribution(NdotH, roughness);
    const double pdf_spec = std::max((D * NdotH) / std::max(4.0 * VdotH, 1e-6), 0.0);
    const double pdf_diff = NdotL / PI;
    const double spec_prob = std::clamp(0.25 + 0.75 * std::clamp(m.metallic, 0.0, 1.0), 0.1, 0.95);
    return (1.0 - spec_prob) * pdf_diff + spec_prob * pdf_spec;
}
static void apply_sand_bump(HitRecord& rec);

// --------------- Integrator (unchanged except toggles wired to CLI) ---------------
template<typename RectT>
Vec3 ray_color(const Ray& r, const Hittable& world, const RectT* area_light, int depth, int max_depth,
               bool PREVIEW, int LIGHT_SAMPLES_PREVIEW, int LIGHT_SAMPLES_FINAL,
               int MAX_DEPTH_PREVIEW, int MAX_DEPTH_FINAL, double SKY_VIEW_GAIN,
               bool show_light_on_primary, bool primary_hit_lighting_only,
               GBuffer* gbuf, int pix_index)
{
    if (depth <= 0) return Vec3(0,0,0);
    HitRecord rec;

    // sky on miss (primary only)
    if (!world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        if (depth == max_depth && gbuf) {
            gbuf->depth[pix_index]    = 1e30f;
            gbuf->normal_x[pix_index] = 0.0f;
            gbuf->normal_y[pix_index] = 0.0f;
            gbuf->normal_z[pix_index] = 0.0f;
        }
        if (depth == max_depth) {
            Vec3 dir = normalize(r.direction);
            return g_hdr_sky ? g_hdr_sky->eval(dir) : g_sky.eval(dir) * SKY_VIEW_GAIN;
        }
        return Vec3(0,0,0);
    }

    // write G-Buffer for primary ray
    if (depth == max_depth && gbuf) {
        uint32_t inst = (rec.hit_object ? rec.hit_object->obj.instance_id : 0);
        gbuf->inst_id[pix_index] = inst;

        float lin_depth = float((rec.point - r.origin).length());
        gbuf->depth[pix_index]    = lin_depth;
        gbuf->normal_x[pix_index] = float(rec.normal.x);
        gbuf->normal_y[pix_index] = float(rec.normal.y);
        gbuf->normal_z[pix_index] = float(rec.normal.z);
    }

    if (!show_light_on_primary && depth == max_depth) {
        if (rec.mat->is_emissive()) {
            if (gbuf) {
                gbuf->inst_id[pix_index]  = 0;
                gbuf->depth[pix_index]    = 1e30f;
                gbuf->normal_x[pix_index] = 0.0f;
                gbuf->normal_y[pix_index] = 0.0f;
                gbuf->normal_z[pix_index] = 0.0f;
            }
            Vec3 dir = normalize(r.direction);
            return g_hdr_sky ? g_hdr_sky->eval(dir) : g_sky.eval(dir) * SKY_VIEW_GAIN;
        }
    }

    apply_sand_bump(rec);

    Vec3 emitted = rec.mat->emitted(rec);

    Vec3 direct(0,0,0);
    const int LIGHT_SAMPLES_PER_HIT = PREVIEW ? LIGHT_SAMPLES_PREVIEW : LIGHT_SAMPLES_FINAL;

    if (area_light) {
        Vec3 L_light(0,0,0);
        const Vec3 V = normalize(-r.direction);
        const double A = area_light->area();
        const Vec3 Le = area_light->mat->emitted(rec);
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

            double pdf_light = dist2 / (cos_l * A);
            Vec3 f = cook_torrance_brdf(*rec.mat, rec.normal, V, wi);
            double pdf_brdf = pbr_brdf_pdf(*rec.mat, rec.normal, V, wi);
            double w = pdf_light / (pdf_light + pdf_brdf);

            L_light += w * Le * f * (cos_i / pdf_light);
        }
        L_light /= double(LIGHT_SAMPLES_PER_HIT);
        direct = L_light;
    }

    if (primary_hit_lighting_only && depth == max_depth) {
        return emitted + direct;
    }

    Ray scattered; Vec3 attenuation;
    if (!rec.mat->scatter(r, rec, attenuation, scattered)) return emitted + direct;

    if (attenuation.x < 1e-3 && attenuation.y < 1e-3 && attenuation.z < 1e-3) return emitted + direct;

    // RR
    if (depth < max_depth - 4) {
        double p = std::max({attenuation.x, attenuation.y, attenuation.z});
        p = std::max(0.05, std::min(1.0, p));
        if (random_double() > p) return emitted + direct;
        attenuation /= p;
    }

    Vec3 indirect = attenuation * ray_color(scattered, world, area_light, depth - 1, max_depth,
                                            PREVIEW, LIGHT_SAMPLES_PREVIEW, LIGHT_SAMPLES_FINAL,
                                            MAX_DEPTH_PREVIEW, MAX_DEPTH_FINAL, SKY_VIEW_GAIN,
                                            show_light_on_primary, primary_hit_lighting_only, gbuf, pix_index);
    return emitted + direct + indirect;
}

// Micro-bump globals (wired by CLI)
static double G_BUMP_FREQ = 5.0;
static double G_BUMP_SCALE = 0.22;

// Slightly modified Lambertian normal perturbation for sand
static void apply_sand_bump(HitRecord& rec) {
    if (rec.mat != sand_ptr && rec.mat != g_terrain_mat) return;
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

struct ForgeCli {
    std::string config_path;
    int frames = 0;

    // Quality overrides (Bun-style CLI authority)
    int width  = -1;
    int height = -1;
    int spp    = -1;

    // HDR / ground / debug overrides
    std::string hdr_path = "";
    double hdr_intensity = -1.0;
    std::string ground_tex = "";
    double ground_scale = -1.0;
    bool verbose = false;

    // Scenario name
    std::string scenario_name = "";
};

static void print_forge_usage() {
    std::cout <<
R"(VisionForge Forge — synthetic dataset generator

Usage:
  visionforge forge --config world.json --frames 100
)";
}

static ForgeCli parse_forge_cli(int argc, char** argv) {
    ForgeCli cli;
    for (int i = 2; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--config" && i + 1 < argc) cli.config_path = argv[++i];
        else if (a == "--frames" && i + 1 < argc) cli.frames = std::stoi(argv[++i]);
        else if (a == "--width" && i + 1 < argc)  cli.width  = std::stoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) cli.height = std::stoi(argv[++i]);
        else if (a == "--spp" && i + 1 < argc)    cli.spp    = std::stoi(argv[++i]);
        // --- New forge-only flags ---
        else if (a == "--hdr" && i + 1 < argc) cli.hdr_path = argv[++i];
        else if (a == "--hdr-intensity" && i + 1 < argc) cli.hdr_intensity = std::stod(argv[++i]);
        else if (a == "--ground-tex" && i + 1 < argc) cli.ground_tex = argv[++i];
        else if (a == "--ground-scale" && i + 1 < argc) cli.ground_scale = std::stod(argv[++i]);
        else if (a == "--verbose") cli.verbose = true;
        else if (a == "--help" || a == "-h") {
            print_forge_usage();
            std::exit(0);
        } else {
            // This is the error you were seeing:
            std::cerr << "Unknown forge flag: " << a << "\n";
            print_forge_usage();
            std::exit(2);
        }
    }
    if (cli.config_path.empty() || cli.frames <= 0) {
        print_forge_usage();
        std::exit(2);
    }
    return cli;
}

static void print_scenario_usage() {
    std::cout <<
R"(VisionForge Scenario — deterministic synthetic generation

Usage:
  visionforge scenario --config world.json --name "MyScenario" --frames 100
)";
}

static ForgeCli parse_scenario_cli(int argc, char** argv) {
    ForgeCli cli;
    for (int i = 2; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--config" && i + 1 < argc) cli.config_path = argv[++i];
        else if (a == "--name" && i + 1 < argc) cli.scenario_name = argv[++i];
        else if (a == "--frames" && i + 1 < argc) cli.frames = std::stoi(argv[++i]);
        else if (a == "--width" && i + 1 < argc)  cli.width  = std::stoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) cli.height = std::stoi(argv[++i]);
        else if (a == "--spp" && i + 1 < argc)    cli.spp    = std::stoi(argv[++i]);
        else if (a == "--hdr" && i + 1 < argc) cli.hdr_path = argv[++i];
        else if (a == "--hdr-intensity" && i + 1 < argc) cli.hdr_intensity = std::stod(argv[++i]);
        else if (a == "--ground-tex" && i + 1 < argc) cli.ground_tex = argv[++i];
        else if (a == "--ground-scale" && i + 1 < argc) cli.ground_scale = std::stod(argv[++i]);
        else if (a == "--verbose") cli.verbose = true;
        else if (a == "--help" || a == "-h") {
            print_scenario_usage();
            std::exit(0);
        } else {
            std::cerr << "Unknown scenario flag: " << a << "\n";
            print_scenario_usage();
            std::exit(2);
        }
    }
    if (cli.config_path.empty() || cli.scenario_name.empty() || cli.frames <= 0) {
        print_scenario_usage();
        std::exit(2);
    }
    return cli;
}


static inline double sample_range(std::mt19937& rng, const ScalarRange& r) {
    std::uniform_real_distribution<double> d(r.min, r.max);
    return d(rng);
}

static inline Vec3 sample_range(std::mt19937& rng, const Vec3Range& r) {
    std::uniform_real_distribution<double> dx(r.min.x, r.max.x);
    std::uniform_real_distribution<double> dy(r.min.y, r.max.y);
    std::uniform_real_distribution<double> dz(r.min.z, r.max.z);
    return Vec3(dx(rng), dy(rng), dz(rng));
}

static int render_frame(const Opts& o,
                        const Camera& cam,
                        const Hittable& world,
                        const std::shared_ptr<YZRect>& rect_light,
                        std::vector<unsigned char>& fb,
                        GBuffer& gbuf,
                        bool primary_hit_lighting_only) {
    bool PREVIEW = o.preview;
    int width = o.width;
    int height = o.height;
    int spp_target = PREVIEW ? o.spp_preview : o.spp;
    int max_depth = PREVIEW ? 6 : o.max_depth;
    int LIGHT_SAMPLES_PREVIEW = o.light_samples_preview;
    int LIGHT_SAMPLES_FINAL = o.light_samples_final;
    int MIN_SPP = o.min_spp;
    double REL_NOISE_TARGET = o.rel_noise_target;

    double F_NUMBER = 2.0;
    double SHUTTER = 1.0 / 30.0;
    int ISO = 400;
    double exposure = exposure_scale(F_NUMBER, SHUTTER, ISO) * o.exposure_comp;

    fb.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
    long long spp_accum = 0;

    if (primary_hit_lighting_only) {
        #if defined(VF_USE_OMP)
          #pragma omp parallel for schedule(dynamic, 1) reduction(+:spp_accum)
        #endif
        for (int j = height - 1; j >= 0; --j) {
            vf_rng::seed_thread_rng(uint64_t(j) * 0x9e3779b97f4a7c15ULL + 42);
            for (int i = 0; i < width; ++i) {
                Vec3 sum(0,0,0);
                int pix_index = ((height-1-j)*width + i);
                for (int s = 0; s < spp_target; ++s) {
                    double u = (i + random_double()) / (width  - 1);
                    double v = (j + random_double()) / (height - 1);
                    Ray r = cam.get_ray(u, v);
                    sum += ray_color(r, world, rect_light.get(), max_depth, max_depth,
                                     PREVIEW, LIGHT_SAMPLES_PREVIEW, LIGHT_SAMPLES_FINAL,
                                     6, o.max_depth, o.sky_gain,
                                     o.show_light_on_primary, true, &gbuf, pix_index);
                }
                spp_accum += spp_target;

                Vec3 pixel = (sum / double(spp_target)) * exposure;
                Vec3 mapped = aces_tonemap(pixel);
                mapped = Vec3(std::sqrt(mapped.x), std::sqrt(mapped.y), std::sqrt(mapped.z));
                int idx = pix_index * 3;
                fb[idx + 0] = (unsigned char)(256 * clamp01(mapped.x));
                fb[idx + 1] = (unsigned char)(256 * clamp01(mapped.y));
                fb[idx + 2] = (unsigned char)(256 * clamp01(mapped.z));
            }
        }
    } else {
        #if defined(VF_USE_OMP)
          #pragma omp parallel for schedule(dynamic, 1) reduction(+:spp_accum)
        #endif
        for (int j = height - 1; j >= 0; --j) {
            vf_rng::seed_thread_rng(uint64_t(j) * 0x9e3779b97f4a7c15ULL + 42);
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
                                       6, o.max_depth, o.sky_gain,
                                       o.show_light_on_primary, false, &gbuf, pix_index);
                    sum += c;

                    double L = luminance(c);
                    double delta = L - meanL;
                    meanL += delta / double(s + 1);
                    M2 += delta * (L - meanL);
                    if (s + 1 >= MIN_SPP) {
                        double var = (s > 0) ? (M2 / s) : 0.0;
                        double sigma = std::sqrt(std::max(0.0, var));
                        double ci95 = 1.96 * sigma / std::sqrt(double(s + 1));
                        if (ci95 < REL_NOISE_TARGET * std::max(1e-3, meanL)) break;
                    }
                }
                spp_accum += (s + 1);

                Vec3 pixel = (sum / double(s + 1)) * exposure;
                Vec3 mapped = aces_tonemap(pixel);
                mapped = Vec3(std::sqrt(mapped.x), std::sqrt(mapped.y), std::sqrt(mapped.z));
                int idx = pix_index * 3;
                fb[idx + 0] = (unsigned char)(256 * clamp01(mapped.x));
                fb[idx + 1] = (unsigned char)(256 * clamp01(mapped.y));
                fb[idx + 2] = (unsigned char)(256 * clamp01(mapped.z));
            }
        }
    }

    return int(double(spp_accum) / double(width * height) + 0.5);
}

// ---- Async I/O pipeline for forge ----

struct FrameObject {
    uint32_t instance_id;
    uint32_t class_id;
    std::string label;
    double x, y, z, yaw;
    double roughness, metallic;
};

struct ForgeFrameData {
    int frame_id;
    int width, height;
    int avg_spp;
    bool is_train;
    std::string split_dir;
    std::string stem;

    std::vector<unsigned char> fb;
    GBuffer gbuf;

    Vec3 lookfrom;
    Vec3 lookat, up;
    double fov;
    double sun_az, sun_el;

    std::vector<FrameObject> objects;

    int spp_target, max_depth;
    unsigned seed;
};

static void write_meta_fast(const std::string& path, const ForgeFrameData& f) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;

    std::fprintf(fp,
R"({"frame_id":%d,"split":"%s","image_width":%d,"image_height":%d,)"
R"("render":{"spp_target":%d,"spp_avg":%d,"max_depth":%d,"seed":%u},)"
R"("camera":{"lookfrom":[%.8g,%.8g,%.8g],"lookat":[%.8g,%.8g,%.8g],)"
R"("up":[%.8g,%.8g,%.8g],"fov_deg":%.8g},)"
R"("sun":{"azimuth_deg":%.8g,"elevation_deg":%.8g},)"
R"("objects":[)",
        f.frame_id, f.is_train ? "train" : "val", f.width, f.height,
        f.spp_target, f.avg_spp, f.max_depth, f.seed,
        f.lookfrom.x, f.lookfrom.y, f.lookfrom.z,
        f.lookat.x, f.lookat.y, f.lookat.z,
        f.up.x, f.up.y, f.up.z, f.fov,
        f.sun_az, f.sun_el);

    for (size_t i = 0; i < f.objects.size(); ++i) {
        const auto& obj = f.objects[i];
        const double yaw_rad = obj.yaw * PI / 180.0;
        const double c = std::cos(yaw_rad), s = std::sin(yaw_rad);
        std::fprintf(fp,
R"({"instance_id":%u,"class_id":%u,"label":"%s",)"
R"("position":[%.8g,%.8g,%.8g],"rotation_y_deg":%.8g,)"
R"("roughness":%.8g,"metallic":%.8g,)"
R"("local_to_world":[[%.8g,0,%.8g,%.8g],[0,1,0,%.8g],[%.8g,0,%.8g,%.8g],[0,0,0,1]]}%s)",
            obj.instance_id, obj.class_id, obj.label.c_str(),
            obj.x, obj.y, obj.z, obj.yaw,
            obj.roughness, obj.metallic,
            c, s, obj.x, obj.y, -s, c, obj.z,
            (i + 1 < f.objects.size()) ? "," : "");
    }
    std::fprintf(fp, "]}\n");
    std::fclose(fp);
}

static void write_yolo_fast(const std::string& path, int W, int H,
                            const std::vector<DetectedBox>& boxes) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    char line[128];
    for (auto& b : boxes) {
        double cx = (b.x0 + b.x1 + 1) * 0.5 / W;
        double cy = (b.y0 + b.y1 + 1) * 0.5 / H;
        double w  = double(b.x1 - b.x0 + 1) / W;
        double h  = double(b.y1 - b.y0 + 1) / H;
        int n = std::snprintf(line, sizeof(line), "%u %.6f %.6f %.6f %.6f\n",
                              b.class_id, cx, cy, w, h);
        std::fwrite(line, 1, n, fp);
    }
    std::fclose(fp);
}

class ForgeIOWorker {
public:
    ForgeIOWorker() : done_(false), error_(false) {
        thread_ = std::thread([this]{ run(); });
    }
    ~ForgeIOWorker() { drain(); }

    void push(ForgeFrameData&& fd) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(std::move(fd));
        }
        cv_.notify_one();
    }

    void drain() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    bool had_error() const { return error_.load(); }

    CocoWriter& coco() { return coco_; }

private:
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<ForgeFrameData> queue_;
    bool done_;
    std::atomic<bool> error_;
    CocoWriter coco_;

    void run() {
        while (true) {
            ForgeFrameData fd;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]{ return !queue_.empty() || done_; });
                if (queue_.empty() && done_) break;
                fd = std::move(queue_.front());
                queue_.pop();
            }
            process(fd);
        }
    }

    void process(const ForgeFrameData& f) {
        const std::string png_path  = f.split_dir + "/" + f.stem + ".png";
        const std::string exr_path  = f.split_dir + "/" + f.stem + "_spatial.exr";
        const std::string meta_path = f.split_dir + "/" + f.stem + "_meta.json";
        const std::string yolo_path = f.split_dir + "/" + f.stem + ".txt";

        if (!vf::write_png_rgb8(png_path.c_str(), f.width, f.height, f.fb.data())) {
            std::cerr << "Forge IO: failed PNG: " << png_path << "\n";
            error_ = true; return;
        }
        if (!vf::write_gbuffer_exr(exr_path.c_str(), f.gbuf)) {
            std::cerr << "Forge IO: failed EXR: " << exr_path << "\n";
            error_ = true; return;
        }

        write_meta_fast(meta_path, f);

        IdRegistry reg;
        for (const auto& obj : f.objects) {
            reg.id_to_label[obj.instance_id] = obj.label;
            reg.id_to_class[obj.instance_id] = obj.class_id;
        }
        auto boxes = boxes_from_mask(f.gbuf, reg);

        write_yolo_fast(yolo_path, f.width, f.height, boxes);

        int coco_img_id = coco_.add_image(f.stem + ".png", f.width, f.height);
        for (auto& b : boxes) {
            coco_.ensure_category(b.class_id, b.label);
            coco_.add_box(coco_img_id, b.class_id, b.x0, b.y0, b.x1, b.y1,
                          b.instance_id, b.label);
        }
    }
};

static void write_coco_fast(const std::string& path, const CocoWriter& coco) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    std::fprintf(fp, "{\"images\":[\n");
    for (size_t i = 0; i < coco.images.size(); ++i) {
        auto& im = coco.images[i];
        std::fprintf(fp, "{\"id\":%d,\"file_name\":\"%s\",\"width\":%d,\"height\":%d}%s\n",
                     im.id, im.file_name.c_str(), im.width, im.height,
                     (i + 1 < coco.images.size()) ? "," : "");
    }
    std::fprintf(fp, "],\"annotations\":[\n");
    for (size_t i = 0; i < coco.anns.size(); ++i) {
        auto& a = coco.anns[i];
        std::fprintf(fp, "{\"id\":%d,\"image_id\":%d,\"category_id\":%d,"
                     "\"bbox\":[%.1f,%.1f,%.1f,%.1f],\"iscrowd\":0,\"area\":%d,"
                     "\"instance_id\":%d,\"label\":\"%s\"}%s\n",
                     a.id, a.image_id, a.category_id,
                     a.x, a.y, a.w, a.h, a.area,
                     a.instance_id, a.label.c_str(),
                     (i + 1 < coco.anns.size()) ? "," : "");
    }
    std::fprintf(fp, "],\"categories\":[\n");
    for (size_t i = 0; i < coco.cats.size(); ++i) {
        auto& c = coco.cats[i];
        std::fprintf(fp, "{\"id\":%d,\"name\":\"%s\"}%s\n",
                     c.id, c.name.c_str(),
                     (i + 1 < coco.cats.size()) ? "," : "");
    }
    std::fprintf(fp, "]}\n");
    std::fclose(fp);
}

static int run_forge_subcommand(int argc, char** argv) {
    ForgeCli cli = parse_forge_cli(argc, argv);
    WorldConfig cfg = load_world_config(cli.config_path, {.strict = true});

    // --- Bun-style CLI Overrides ---
    // These allow terminal flags to win over world.json values
    if (cli.width  > 0) cfg.render.width  = cli.width;
    if (cli.height > 0) cfg.render.height = cli.height;
    if (cli.spp    > 0) cfg.render.spp    = cli.spp;

    if (!cli.hdr_path.empty()) cfg.hdr_path = cli.hdr_path;
    if (cli.hdr_intensity > 0) cfg.hdr_intensity = cli.hdr_intensity;
    if (!cli.ground_tex.empty()) cfg.ground_tex = cli.ground_tex;
    if (cli.ground_scale > 0) cfg.ground_scale = cli.ground_scale;

    if (cli.verbose) {
        std::cout << "[FORGE] Environment: " << (cfg.hdr_path.empty() ? "Analytic Sky" : cfg.hdr_path) << "\n"
                  << "[FORGE] Ground Tex:  " << (cfg.ground_tex.empty() ? "Flat Color" : cfg.ground_tex) << "\n";
    }

    std::filesystem::create_directories(cfg.dataset.root);
    const std::string train_dir = cfg.dataset.root + "/train";
    const std::string val_dir = cfg.dataset.root + "/val";
    std::filesystem::create_directories(train_dir);
    std::filesystem::create_directories(val_dir);

    Opts o;
    o.width         = cfg.render.width;
    o.height        = cfg.render.height;
    o.spp           = cfg.render.spp;
    o.max_depth     = cfg.render.max_depth;
    o.exposure_comp = cfg.render.exposure;
    o.sky_gain      = cfg.render.sky_gain;
    o.preview       = cfg.render.preview;
    o.seed          = cfg.render.seed;
    o.write_exr     = true;

    // Adaptive sampling guardrails for forge:
    // - keep min_spp reasonably high so dark regions converge
    // - relax noise target for speed (dataset focus)
    o.min_spp          = 8;
    o.rel_noise_target = 0.1;
    o.light_samples_final = 1;

    // --- Phase 6: Global Resource Initialization ---
    if (!cfg.hdr_path.empty()) {
        g_hdr_sky = std::make_shared<HDRSky>();
        if (!g_hdr_sky->load(cfg.hdr_path, cfg.hdr_intensity)) {
            std::cerr << "Forge: failed to load HDR sky: " << cfg.hdr_path << "\n";
            g_hdr_sky.reset();
        }
    }

    if (!cfg.ground_tex.empty()) {
        auto img = std::make_shared<ImageTexture>();
        if (img->load(cfg.ground_tex)) {
            g_terrain_mat = std::make_shared<TriplanarMaterial>(img, cfg.ground_scale);
        } else {
            std::cerr << "Forge: failed to load ground texture: " << cfg.ground_tex << "\n";
        }
    }

    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    HeightField hf(cfg.terrain.xmin, cfg.terrain.xmax,
                   cfg.terrain.zmin, cfg.terrain.zmax,
                   cfg.terrain.nx, cfg.terrain.nz,
                   cfg.terrain.amp, cfg.terrain.scale,
                   cfg.render.seed);
    hf.generate();

    HittableList static_world;
    auto sun_em = std::make_shared<DiffuseLight>(o.light_color, o.light_intensity);
    auto rect_light = std::make_shared<YZRect>(o.light_y0, o.light_y1, o.light_z0, o.light_z1, o.light_x, sun_em);
    static_world.add(rect_light);
    
    // Use triplanar if available, else standard sand
    hf.to_triangles(static_world, g_terrain_mat ? g_terrain_mat : sand_ptr);
    
    std::vector<std::shared_ptr<Hittable>> static_objs = static_world.objects;
    auto static_bvh = std::make_shared<BVHNode>(static_objs, 0, static_objs.size());

    AssetManager asset_mgr;
    if (!asset_mgr.load_all(cfg.assets, parse_color_one)) {
        std::cerr << "Forge: failed to load assets\n";
        return 2;
    }

    std::mt19937 dr_rng(cfg.render.seed ^ 0x9e3779b1u);
    std::uniform_real_distribution<double> sink_rng(0.01, 0.05); // Realistic sink range
    int train_count = static_cast<int>(std::round(cli.frames * cfg.dataset.train_split));

    const double aspect = double(o.width) / double(o.height);
    std::vector<unsigned char> fb;
    fb.reserve(static_cast<size_t>(o.width) * static_cast<size_t>(o.height) * 3);
    GBuffer gbuf(o.width, o.height);

    HittableList frame_world;
    frame_world.objects.reserve(2);

    ForgeIOWorker io_worker;
    double total_render_ms = 0.0;

    for (int frame = 0; frame < cli.frames; ++frame) {
        Vec3 lookfrom = sample_range(dr_rng, cfg.camera.lookfrom);
        double fov = sample_range(dr_rng, cfg.camera.fov_deg);
        double sun_az = sample_range(dr_rng, cfg.lighting.sun_azimuth_deg);
        double sun_el = sample_range(dr_rng, cfg.lighting.sun_elevation_deg);
        double obj_x = sample_range(dr_rng, cfg.placement.x);
        double obj_z = sample_range(dr_rng, cfg.placement.z);
        double obj_yaw = sample_range(dr_rng, cfg.placement.yaw_deg);
        
        auto terrain_sample = hf.get_terrain_height_and_normal(obj_x, obj_z);
        auto& asset = asset_mgr.select(dr_rng);
        
        double obj_roughness = sample_range(dr_rng, asset.config.roughness);
        double obj_metallic = sample_range(dr_rng, asset.config.metallic);
        asset.material->set_parameters(asset.material->base_color, obj_roughness, obj_metallic);

        g_sky.set_angles(sun_az, sun_el);

        // --- Phase 5: Grounding Math Using SceneNode ---
        auto node = std::make_shared<SceneNode>(asset.name);
        node->object = asset.mesh;
        node->class_id = asset.config.class_id;
        node->label = asset.config.label;
        node->instance_id = 1;
        node->local_transform.position = Vec3(obj_x, 0, obj_z);
        node->local_transform.rotation = Vec3(0, obj_yaw, 0); // Yaw maps to Y in rotation convention for flat_pack
        node->local_transform.scale = Vec3(asset.config.scale, asset.config.scale, asset.config.scale);
        node->grounding_constraint = true;
        
        double frame_sink = sink_rng(dr_rng);
        node->y_offset = asset.config.y_offset - frame_sink;

        node->update_transforms();
        node->apply_grounding(hf);
        
        frame_world.objects.clear();
        frame_world.objects.push_back(static_bvh);
        node->flat_pack(frame_world);

        double obj_y = node->world_transform.position.y;

        // --- Mathematical Verification Log ---
        if (cli.verbose) {
            std::printf("[FRAME %04d] Asset: %s | Snap-Y: %.2f | Normal: (%.2f, %.2f, %.2f) | Sink: %.2f\n",
                        frame,
                        asset.config.label.c_str(),
                        obj_y,
                        terrain_sample.normal.x,
                        terrain_sample.normal.y,
                        terrain_sample.normal.z,
                        frame_sink);
        }


        const double focus_dist = (lookfrom - cfg.camera.lookat).length();
        Camera cam(lookfrom, cfg.camera.lookat, cfg.camera.up, fov, aspect, 0.0, focus_dist, 0.0, 1.0);

        vf_rng::seed_thread_rng(uint64_t(o.seed) + uint64_t(frame));
        gbuf.clear();
        auto t_render_start = std::chrono::high_resolution_clock::now();
        int avg_spp = render_frame(o, cam, frame_world, rect_light, fb, gbuf, true);
        auto t_render_end = std::chrono::high_resolution_clock::now();
        
        double render_ms = std::chrono::duration<double, std::milli>(t_render_end - t_render_start).count();
        total_render_ms += render_ms;

        const bool is_train = (frame < train_count);
        const std::string& split_dir = is_train ? train_dir : val_dir;
        char stem_buf[32];
        std::snprintf(stem_buf, sizeof(stem_buf), "frame_%04d", frame);

        ForgeFrameData fd;
        fd.frame_id = frame; fd.width = o.width; fd.height = o.height;
        fd.avg_spp = avg_spp; fd.is_train = is_train; fd.split_dir = split_dir; fd.stem = stem_buf;
        fd.fb = std::move(fb); fd.gbuf = std::move(gbuf);
        fd.lookfrom = lookfrom; fd.lookat = cfg.camera.lookat; fd.up = cfg.camera.up; fd.fov = fov;
        fd.sun_az = sun_az; fd.sun_el = sun_el;

        FrameObject fo;
        fo.instance_id = 1;
        fo.class_id = asset.config.class_id;
        fo.label = asset.config.label;
        fo.x = obj_x; fo.y = obj_y; fo.z = obj_z; fo.yaw = obj_yaw;
        fo.roughness = obj_roughness; fo.metallic = obj_metallic;
        fd.objects.push_back(fo);

        fd.spp_target = o.spp; fd.max_depth = o.max_depth; fd.seed = o.seed + (unsigned)frame;

        io_worker.push(std::move(fd));

        fb.clear();
        fb.reserve(static_cast<size_t>(o.width) * static_cast<size_t>(o.height) * 3);
        gbuf = GBuffer(o.width, o.height);

        if (cli.verbose) {
            std::cout << "Frame " << (frame + 1) << "/" << cli.frames << "  render=" << (int)render_ms << "ms\n";
        }
    }

    std::cout << "Rendering done. Avg render: " << (int)(total_render_ms / cli.frames) << "ms/frame. Flushing I/O...\n";
    io_worker.drain();
    write_coco_fast(cfg.dataset.root + "/annotations_coco.json", io_worker.coco());

    return 0;
}

static int run_scenario_subcommand(int argc, char** argv) {
    ForgeCli cli = parse_scenario_cli(argc, argv);
    WorldConfig cfg = load_world_config(cli.config_path, {.strict = true});

    // Apply basic flags as run_forge does
    if (cli.width  > 0) cfg.render.width  = cli.width;
    if (cli.height > 0) cfg.render.height = cli.height;
    if (cli.spp    > 0) cfg.render.spp    = cli.spp;
    if (!cli.hdr_path.empty()) cfg.hdr_path = cli.hdr_path;
    if (cli.hdr_intensity > 0) cfg.hdr_intensity = cli.hdr_intensity;
    if (!cli.ground_tex.empty()) cfg.ground_tex = cli.ground_tex;
    if (cli.ground_scale > 0) cfg.ground_scale = cli.ground_scale;

    WorldConfig::Scenario* active_scenario = nullptr;
    for (auto& s : cfg.scenarios) {
        if (s.name == cli.scenario_name) {
            active_scenario = &s;
            break;
        }
    }
    if (!active_scenario) {
        std::cerr << "Scenario '" << cli.scenario_name << "' not found in config.\n";
        return 1;
    }

    std::filesystem::create_directories(cfg.dataset.root);
    const std::string train_dir = cfg.dataset.root + "/train";
    const std::string val_dir = cfg.dataset.root + "/val";
    std::filesystem::create_directories(train_dir);
    std::filesystem::create_directories(val_dir);

    Opts o;
    o.width         = cfg.render.width;
    o.height        = cfg.render.height;
    o.spp           = cfg.render.spp;
    o.max_depth     = cfg.render.max_depth;
    o.exposure_comp = cfg.render.exposure;
    o.sky_gain      = cfg.render.sky_gain;
    o.preview       = cfg.render.preview;
    o.seed          = cfg.render.seed;
    o.write_exr     = true;
    o.min_spp          = 8;
    o.rel_noise_target = 0.1;
    o.light_samples_final = 1;

    // Load resources
    if (!cfg.hdr_path.empty()) {
        g_hdr_sky = std::make_shared<HDRSky>();
        if (!g_hdr_sky->load(cfg.hdr_path, cfg.hdr_intensity)) g_hdr_sky.reset();
    }
    if (!cfg.ground_tex.empty()) {
        auto img = std::make_shared<ImageTexture>();
        if (img->load(cfg.ground_tex)) g_terrain_mat = std::make_shared<TriplanarMaterial>(img, cfg.ground_scale);
    }

    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    HeightField hf(cfg.terrain.xmin, cfg.terrain.xmax, cfg.terrain.zmin, cfg.terrain.zmax,
                   cfg.terrain.nx, cfg.terrain.nz, cfg.terrain.amp, cfg.terrain.scale, cfg.render.seed);
    hf.generate();

    HittableList static_world;
    auto sun_em = std::make_shared<DiffuseLight>(o.light_color, o.light_intensity);
    auto rect_light = std::make_shared<YZRect>(o.light_y0, o.light_y1, o.light_z0, o.light_z1, o.light_x, sun_em);
    static_world.add(rect_light);
    hf.to_triangles(static_world, g_terrain_mat ? g_terrain_mat : sand_ptr);
    
    std::vector<std::shared_ptr<Hittable>> static_objs = static_world.objects;
    auto static_bvh = std::make_shared<BVHNode>(static_objs, 0, static_objs.size());

    AssetManager asset_mgr;
    if (!asset_mgr.load_all(cfg.assets, parse_color_one)) return 2;

    std::mt19937 dr_rng(cfg.render.seed ^ 0xabcdef12u);
    int train_count = static_cast<int>(std::round(cli.frames * cfg.dataset.train_split));

    const double aspect = double(o.width) / double(o.height);
    std::vector<unsigned char> fb;
    fb.reserve(static_cast<size_t>(o.width) * static_cast<size_t>(o.height) * 3);
    GBuffer gbuf(o.width, o.height);
    HittableList frame_world;
    ForgeIOWorker io_worker;
    double total_render_ms = 0.0;
    
    // Scene node builder
    std::function<std::shared_ptr<SceneNode>(const WorldConfig::NodeEntry&)> build_node;
    build_node = [&](const WorldConfig::NodeEntry& entry) -> std::shared_ptr<SceneNode> {
        auto n = std::make_shared<SceneNode>(entry.name);
        n->local_transform.position = entry.position;
        n->local_transform.rotation = entry.rotation;
        n->local_transform.scale = entry.scale;
        n->grounding_constraint = entry.grounding_constraint;
        n->y_offset = entry.y_offset;
        if (!entry.asset.empty()) {
            LoadedAsset* loaded = asset_mgr.find_by_name(entry.asset);
            if (loaded) {
                n->object = loaded->mesh;
                n->class_id = loaded->config.class_id;
                n->label = loaded->config.label;
            }
        }
        for (const auto& c : entry.children) n->add_child(build_node(c));
        return n;
    };

    auto root_node = std::make_shared<SceneNode>("root");
    for (const auto& ne : active_scenario->root_nodes) {
        root_node->add_child(build_node(ne));
    }

    for (int frame = 0; frame < cli.frames; ++frame) {
        double sun_az = sample_range(dr_rng, cfg.lighting.sun_azimuth_deg);
        double sun_el = sample_range(dr_rng, cfg.lighting.sun_elevation_deg);
        g_sky.set_angles(sun_az, sun_el);

        for (size_t i=0; i<asset_mgr.size(); ++i) {
            auto& asset = asset_mgr[i];
            double dr_r = sample_range(dr_rng, asset.config.roughness);
            double dr_m = sample_range(dr_rng, asset.config.metallic);
            asset.material->set_parameters(asset.material->base_color, dr_r, dr_m);
        }

        root_node->update_transforms();
        root_node->apply_grounding(hf);
        
        frame_world.objects.clear();
        frame_world.objects.push_back(static_bvh);
        
        uint32_t inst_counter = 1;
        std::vector<FrameObject> frame_objs;
        std::function<void(std::shared_ptr<SceneNode>)> prep_for_frame = [&](std::shared_ptr<SceneNode> n) {
            if (n->object) {
                n->instance_id = inst_counter++;
                FrameObject fo;
                fo.instance_id = n->instance_id;
                fo.class_id = n->class_id;
                fo.label = n->label;
                fo.x = n->world_transform.position.x;
                fo.z = n->world_transform.position.z;
                fo.yaw = n->world_transform.rotation.y;
                fo.y = n->world_transform.position.y;
                
                double rough = 0.5, metal = 0.0;
                if (!n->label.empty() || !n->name.empty()) {
                    auto* la = asset_mgr.find_by_name(n->label.empty() ? n->name : n->label);
                    if (la) {
                        rough = la->material->roughness;
                        metal = la->material->metallic;
                    }
                }
                fo.roughness = rough;
                fo.metallic = metal;
                frame_objs.push_back(fo);
            }
            for (auto c : n->children) prep_for_frame(c);
        };
        prep_for_frame(root_node);

        root_node->flat_pack(frame_world);

        const double focus_dist = (active_scenario->camera.lookfrom.max - active_scenario->camera.lookat).length();
        Vec3 lf = active_scenario->camera.lookfrom.max;
        double c_fov = active_scenario->camera.fov_deg.max;
        Camera cam(lf, active_scenario->camera.lookat, active_scenario->camera.up, c_fov, aspect, 0.0, focus_dist, 0.0, 1.0);

        vf_rng::seed_thread_rng(uint64_t(o.seed) + uint64_t(frame));
        gbuf.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        int avg_spp = render_frame(o, cam, frame_world, rect_light, fb, gbuf, true);
        double render_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        total_render_ms += render_ms;
        if (cli.verbose) std::cout << "Scenario Frame " << (frame + 1) << "/" << cli.frames << " " << (int)render_ms << "ms\n";

        const bool is_train = (frame < train_count);
        char stem_buf[32]; std::snprintf(stem_buf, sizeof(stem_buf), "sfrm_%04d", frame);
        
        ForgeFrameData fd;
        fd.frame_id = frame; fd.width = o.width; fd.height = o.height;
        fd.avg_spp = avg_spp; fd.is_train = is_train; fd.split_dir = is_train ? train_dir : val_dir; fd.stem = stem_buf;
        fd.fb = std::move(fb); fd.gbuf = std::move(gbuf);
        fd.lookfrom = lf; fd.lookat = active_scenario->camera.lookat; fd.up = active_scenario->camera.up; fd.fov = c_fov;
        fd.sun_az = sun_az; fd.sun_el = sun_el;
        fd.objects = std::move(frame_objs);
        fd.spp_target = o.spp; fd.max_depth = o.max_depth; fd.seed = o.seed + (unsigned)frame;
        io_worker.push(std::move(fd));

        fb.clear(); fb.reserve(static_cast<size_t>(o.width) * static_cast<size_t>(o.height) * 3);
        gbuf = GBuffer(o.width, o.height);
    }

    io_worker.drain();
    write_coco_fast(cfg.dataset.root + "/scenario_coco.json", io_worker.coco());
    return 0;
}

// ---------------------------- MAIN ----------------------------
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "forge") {
        return run_forge_subcommand(argc, argv);
    }
    if (argc > 1 && std::string(argv[1]) == "scenario") {
        return run_scenario_subcommand(argc, argv);
    }

    Opts o = parse(argc, argv);
    vf_rng::seed_thread_rng(o.seed);

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

    // sky (CLI)
    g_sky.set_angles(o.sun_az_deg, o.sun_el_deg); // requires small setter; see notes below
    g_sky.set_turbidity(o.turbidity);

    // HDR sky
    if (!o.hdr_path.empty()) {
        g_hdr_sky = std::make_shared<HDRSky>();
        if (!g_hdr_sky->load(o.hdr_path, o.hdr_intensity)) {
            std::cerr << "Warning: failed to load HDR sky: " << o.hdr_path << "\n";
            g_hdr_sky.reset();
        }
    }

    // materials
    sand_ptr = std::make_shared<Lambertian>(Vec3(0.78, 0.72, 0.58));
    auto red_m  = std::make_shared<Lambertian>(Vec3(0.90, 0.18, 0.14));
    auto blu_m  = std::make_shared<Lambertian>(Vec3(0.18, 0.55, 0.95));
    auto grn_m  = std::make_shared<Lambertian>(Vec3(0.22, 0.84, 0.25));
    auto wht_m  = std::make_shared<Lambertian>(Vec3(0.85, 0.85, 0.85));

    // sand bump controls
    G_BUMP_FREQ  = o.sand_bump_freq;
    G_BUMP_SCALE = o.sand_bump_scale;

    // ground texture (triplanar)
    if (!o.ground_tex_path.empty()) {
        auto img = std::make_shared<ImageTexture>();
        if (img->load(o.ground_tex_path)) {
            auto tri_mat = std::make_shared<TriplanarMaterial>(img, o.ground_scale);
            g_terrain_mat = tri_mat;
        } else {
            std::cerr << "Warning: failed to load ground texture, using default sand.\n";
        }
    }

    // terrain
    HeightField hf(o.XMIN, o.XMAX, o.ZMIN, o.ZMAX, o.terrain_nx, o.terrain_nz, o.terrain_amp, o.terrain_scale, /*seed*/1337);
    hf.generate();

    // camera — ensure lookfrom is safely above terrain
    Vec3 lookfrom = o.lookfrom;
    Vec3 lookat   = o.lookat;
    double vfov_deg = o.fov_deg;
    {
        double cam_ground = hf.height_at(lookfrom.x, lookfrom.z);
        double cam_min_y  = cam_ground + o.terrain_amp + 2.0;
        if (lookfrom.y < cam_min_y) lookfrom.y = cam_min_y;
    }
    double focus_dist = (lookfrom - lookat).length();
    Camera cam(lookfrom, lookat, Vec3(0,1,0), vfov_deg, aspect, 0.0, focus_dist, 0.0, 1.0);

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
    std::vector<std::shared_ptr<Material>> cube_mats;
    cube_mats.reserve(cube_colors.size());
    for (auto& c : cube_colors) cube_mats.push_back(std::make_shared<PBRMaterial>(c, 0.65, 0.0));

    // cube placement
    struct CubePose { Vec3 c; double edge; double roll; double pitch; int color_idx; std::string label; Vec3 terrain_normal; };
    std::vector<CubePose> cubes; cubes.reserve(o.cubes);
    std::mt19937 rng(o.seed ^ 0x9e3779b1);
    std::uniform_real_distribution<double> edge_rng(o.cube_edge_min, o.cube_edge_max);
    std::uniform_real_distribution<double> tilt_rng(-o.cube_tilt_abs, o.cube_tilt_abs);

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
                auto sample = hf.get_terrain_height_and_normal(x, z);
                hf.carve_footprint(x, z, e, 0.0, 0.10 * e);
                int ci = count % (int)cube_mats.size();
                cubes.push_back({Vec3(x, sample.height, z), e, tilt_rng(rng), tilt_rng(rng), ci, color_labels[ci], sample.normal});
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
            auto sample = hf.get_terrain_height_and_normal(x, z);
            hf.carve_footprint(x, z, e, 0.0, 0.10 * e);
            int ci = (int)cubes.size() % (int)cube_mats.size();
            cubes.push_back({Vec3(x, sample.height, z), e, tilt_rng(rng), tilt_rng(rng), ci, color_labels[ci], sample.normal});
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
        auto aligned = std::make_shared<SlopeAlign>(tilted, cp.terrain_normal);

        AABB aabb;
        aligned->bounding_box(aabb);
        cp.c.y = snap_y(cp.c.y, aabb.max().y - aabb.min().y);

        auto positioned = std::make_shared<Translate>(aligned, cp.c);

        uint32_t id = next_instance_id++;
        std::string label = cp.label.empty()? "cube" : cp.label;
        ObjectInfo info{ id, /*class_id=*/1u, label.c_str() };
        auto tagged = std::make_shared<Tag>(positioned, info);
        objects.add(tagged);

        reg.id_to_label[id] = label;
        reg.id_to_class[id] = 1u;
    }

    // OBJ mesh loading — snap to terrain + slope alignment + sim-to-real sink
    if (!o.obj_path.empty()) {
        bool color_ok = true;
        std::string obj_label;
        Vec3 obj_col = parse_color_one(o.obj_color, color_ok, obj_label);
        auto obj_mat = std::make_shared<PBRMaterial>(obj_col, 0.5, 0.0);

        auto mesh = Mesh::from_obj(o.obj_path, obj_mat, Vec3(0,0,0), o.obj_scale);
        if (mesh) {
            auto sample = hf.get_terrain_height_and_normal(o.obj_pos.x, o.obj_pos.z);
            auto aligned = std::make_shared<SlopeAlign>(mesh, sample.normal);

            AABB aabb;
            aligned->bounding_box(aabb);
            double obj_y = snap_y(sample.height, aabb.max().y - aabb.min().y, o.obj_sink);
            auto positioned = std::make_shared<Translate>(aligned, Vec3(o.obj_pos.x, obj_y, o.obj_pos.z));

            std::fprintf(stderr,
                "[GROUNDING] Pos: (%.2f, %.2f) | Terrain H: %.2f | Snap-Y: %.2f"
                " | Normal: (%.2f, %.2f, %.2f) | Sink: %.2f\n",
                o.obj_pos.x, o.obj_pos.z,
                sample.height, obj_y,
                sample.normal.x, sample.normal.y, sample.normal.z,
                o.obj_sink);

            constexpr uint32_t OBJ_INSTANCE_ID = 100;
            if (obj_label.empty()) obj_label = "mesh";
            ObjectInfo info{ OBJ_INSTANCE_ID, /*class_id=*/2u, obj_label.c_str() };
            auto tagged = std::make_shared<Tag>(positioned, info);
            objects.add(tagged);

            reg.id_to_label[OBJ_INSTANCE_ID] = obj_label;
            reg.id_to_class[OBJ_INSTANCE_ID] = 2u;
        }
    }

    // tessellate sand (use triplanar material if available)
    hf.to_triangles(objects, g_terrain_mat ? g_terrain_mat : sand_ptr);

    // BVH
    std::vector<std::shared_ptr<Hittable>> objs_vec = objects.objects;
    BVHNode world(objs_vec, 0, objs_vec.size());

    // render
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> fb(width * height * 3, 0);
    GBuffer gbuf(width, height);
    double exposure = exposure_scale(F_NUMBER, SHUTTER, ISO) * EXPOSURE_COMP;

    long long spp_accum = 0;

    if (o.depth_only) {
        // Fast primary-ray-only pass: 1 spp, no shading -- ground-truth depth & normals
        #if defined(VF_USE_OMP)
          #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int j = height - 1; j >= 0; --j) {
            for (int i = 0; i < width; ++i) {
                int pix_index = ((height-1-j)*width + i);
                double u = (i + 0.5) / (width  - 1);
                double v = (j + 0.5) / (height - 1);
                Ray r = cam.get_ray(u, v);
                HitRecord rec;
                if (world.hit(r, 0.001, std::numeric_limits<double>::infinity(), rec)) {
                    bool is_hidden_light = !o.show_light_on_primary
                                           && rec.mat->is_emissive();
                    if (is_hidden_light) {
                        gbuf.depth[pix_index]    = 1e30f;
                        gbuf.normal_x[pix_index] = 0.0f;
                        gbuf.normal_y[pix_index] = 0.0f;
                        gbuf.normal_z[pix_index] = 0.0f;
                    } else {
                        gbuf.inst_id[pix_index]  = rec.hit_object ? rec.hit_object->obj.instance_id : 0;
                        gbuf.depth[pix_index]    = float((rec.point - r.origin).length());
                        gbuf.normal_x[pix_index] = float(rec.normal.x);
                        gbuf.normal_y[pix_index] = float(rec.normal.y);
                        gbuf.normal_z[pix_index] = float(rec.normal.z);
                    }
                } else {
                    gbuf.depth[pix_index]    = 1e30f;
                    gbuf.normal_x[pix_index] = 0.0f;
                    gbuf.normal_y[pix_index] = 0.0f;
                    gbuf.normal_z[pix_index] = 0.0f;
                }
            }
        }
        spp_accum = (long long)width * height;
    } else {
        #if defined(VF_USE_OMP)
          #pragma omp parallel for schedule(dynamic, 1) reduction(+:spp_accum)
        #endif
        for (int j = height - 1; j >= 0; --j) {
            vf_rng::seed_thread_rng(uint64_t(j) * 0x9e3779b97f4a7c15ULL + o.seed);
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
                                       o.show_light_on_primary, /*primary_hit_lighting_only=*/false, &gbuf, pix_index);
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
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double render_secs = std::chrono::duration<double>(t1 - t0).count();
    int avg_spp = int(double(spp_accum) / double(width*height) + 0.5);

    bool single_file = (o.out.size() >= 4 &&
                        (o.out.substr(o.out.size()-4) == ".ppm" ||
                         o.out.substr(o.out.size()-4) == ".png"));

    if (single_file) {
        if (o.write_bbox_overlay && !o.depth_only) {
            auto sf_boxes = boxes_from_mask(gbuf, reg);
            for (auto& b : sf_boxes)
                draw_rect(fb, width, height, b.x0, b.y0, b.x1, b.y1);
        }
        bool is_png = o.out.substr(o.out.size()-4) == ".png";
        if (is_png) {
            vf::write_png_rgb8(o.out.c_str(), width, height, fb.data());
        } else {
            FILE* fp = std::fopen(o.out.c_str(), "wb");
            if (fp) {
                std::fprintf(fp, "P6\n%d %d\n255\n", width, height);
                std::fwrite(fb.data(), 1, fb.size(), fp);
                std::fclose(fp);
            }
        }
        std::cout << "Wrote " << o.out << "  render=" << render_secs << "s  avg_spp=" << avg_spp << "\n";
        return 0;
    }

    // ---- Directory output mode: full forge-style labels ----
    std::filesystem::create_directories(o.out);

    std::vector<DetectedBox> tight;
    if (!o.depth_only) {
        tight = boxes_from_mask(gbuf, reg);
        if (o.write_bbox_overlay) {
            for (auto& b : tight)
                draw_rect(fb, width, height, b.x0, b.y0, b.x1, b.y1);
        }
    }

    std::cout << "Render done: " << render_secs << "s  avg_spp=" << avg_spp << ". Writing outputs...\n";

    auto t_io_start = std::chrono::high_resolution_clock::now();

    FILE* ppm_fp = std::fopen((o.out + "/image.ppm").c_str(), "wb");
    if (ppm_fp) {
        std::fprintf(ppm_fp, "P6\n%d %d\n255\n", width, height);
        std::fwrite(fb.data(), 1, fb.size(), ppm_fp);
        std::fclose(ppm_fp);
    }

    vf::write_png_rgb8((o.out + "/image.png").c_str(), width, height, fb.data());

    if (o.write_exr || o.depth_only) {
        vf::write_gbuffer_exr((o.out + "/gbuffer.exr").c_str(), gbuf);
    }

    if (!o.depth_only) {

        // inst.pgm (reuse the same gbuf scan we already have)
        {
            FILE* pgm_fp = std::fopen((o.out + "/inst.pgm").c_str(), "wb");
            if (pgm_fp) {
                std::fprintf(pgm_fp, "P5\n%d %d\n255\n", width, height);
                const size_t npix = static_cast<size_t>(width) * height;
                std::vector<unsigned char> pgm_row(npix);
                for (size_t i = 0; i < npix; ++i)
                    pgm_row[i] = static_cast<unsigned char>(gbuf.inst_id[i] > 255 ? 255 : gbuf.inst_id[i]);
                std::fwrite(pgm_row.data(), 1, npix, pgm_fp);
                std::fclose(pgm_fp);
            }
        }

        // YOLO (fast-path)
        write_yolo_fast(o.out + "/labels_yolo.txt", width, height, tight);

        // COCO (fast-path)
        {
            CocoWriter coco;
            int img_id = coco.add_image("image.png", width, height);
            for (auto& b : tight) {
                coco.ensure_category(b.class_id, b.label);
                coco.add_box(img_id, b.class_id, b.x0, b.y0, b.x1, b.y1, b.instance_id, b.label);
            }
            write_coco_fast(o.out + "/labels_coco.json", coco);
        }

        // Bounding box CSV + JSON (fast-path, single FILE*)
        {
            FILE* csv_fp = std::fopen((o.out + "/bboxes.csv").c_str(), "wb");
            if (csv_fp) {
                std::fprintf(csv_fp, "instance_id,class_id,label,xmin,ymin,xmax,ymax,width,height\n");
                for (auto& b : tight)
                    std::fprintf(csv_fp, "%u,%u,%s,%d,%d,%d,%d,%d,%d\n",
                                 b.instance_id, b.class_id, b.label.c_str(),
                                 b.x0, b.y0, b.x1, b.y1, width, height);
                std::fclose(csv_fp);
            }
        }
        {
            FILE* json_fp = std::fopen((o.out + "/bboxes.json").c_str(), "wb");
            if (json_fp) {
                std::fprintf(json_fp, "{\"image_width\":%d,\"image_height\":%d,\"boxes\":[\n", width, height);
                for (size_t i = 0; i < tight.size(); ++i) {
                    auto& b = tight[i];
                    std::fprintf(json_fp, "{\"instance_id\":%u,\"class_id\":%u,\"label\":\"%s\","
                                 "\"xmin\":%d,\"ymin\":%d,\"xmax\":%d,\"ymax\":%d}%s\n",
                                 b.instance_id, b.class_id, b.label.c_str(),
                                 b.x0, b.y0, b.x1, b.y1,
                                 (i + 1 < tight.size()) ? "," : "");
                }
                std::fprintf(json_fp, "]}\n");
                std::fclose(json_fp);
            }
        }
    }

    write_manifest(o.out, o, avg_spp, render_secs);

    auto t_io_end = std::chrono::high_resolution_clock::now();
    double io_secs = std::chrono::duration<double>(t_io_end - t_io_start).count();
    std::cout << "Done. render=" << render_secs << "s  io=" << io_secs << "s  avg_spp=" << avg_spp << "\n";
    return 0;
}
