#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "visionforge/vec3.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// 2D box in pixels
struct Box2D {
    int x0=0, y0=0, x1=0, y1=0;
    std::string label;
    bool valid=false;
};

// Pose + label for a cube; used for projecting its bbox.
struct CubePose {
    Vec3 center;
    double edge = 1.0;
    double rollZ = 0.0;   // degrees
    double pitchX = 0.0;  // degrees
    std::string label;
};

struct CameraBasis {
    Vec3 origin, u, v, w;
    double vfov_deg = 45.0;
    double aspect = 1.7777778;
};

inline CameraBasis make_camera_basis(const Vec3& lookfrom, const Vec3& lookat, const Vec3& vup,
                                     double vfov_deg, double aspect)
{
    CameraBasis cb;
    cb.origin = lookfrom;
    Vec3 w   = normalize(lookfrom - lookat);     // w points backward (RTOW convention)
    Vec3 u   = normalize(cross(vup, w));
    Vec3 v   = cross(w, u);
    cb.u = u; cb.v = v; cb.w = w;
    cb.vfov_deg = vfov_deg; cb.aspect = aspect;
    return cb;
}

// Exact projection matching the camera math used by your renderer.
inline bool project_from_camera(const CameraBasis& cb, const Vec3& P,
                                int width, int height, int& out_x, int& out_y)
{
    Vec3 pc = P - cb.origin;
    double X = dot(pc, cb.u);
    double Y = dot(pc, cb.v);
    double Z = dot(pc, -cb.w);              // forward
    if (Z <= 1e-9) return false;

    double half_h = std::tan((cb.vfov_deg * PI/180.0) * 0.5);
    double half_w = cb.aspect * half_h;

    double nx = X / (Z * half_w);
    double ny = Y / (Z * half_h);

    double sx = 0.5 * (nx + 1.0);
    double sy = 0.5 * (1.0 - ny);

    if (!(sx == sx && sy == sy)) return false; // NaN
    out_x = (int)std::round(std::clamp(sx, 0.0, 1.0) * (width  - 1));
    out_y = (int)std::round(std::clamp(sy, 0.0, 1.0) * (height - 1));
    return true;
}

// Small helpers for rotating cube corners (degrees).
inline Vec3 rotateX_pt(Vec3 p, double deg){
    double r = deg * PI/180.0, s=std::sin(r), c=std::cos(r);
    return Vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z);
}
inline Vec3 rotateZ_pt(Vec3 p, double deg){
    double r = deg * PI/180.0, s=std::sin(r), c=std::cos(r);
    return Vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z);
}

// Tight 2D bbox around a rotated cube (using camera projection above).
inline Box2D cube_bbox_screen_rot(const CameraBasis& cb, const CubePose& P,
                                  int width, int height)
{
    const double h = P.edge * 0.5;
    const Vec3 corners[8] = {
        {-h,-h,-h},{+h,-h,-h},{-h,+h,-h},{+h,+h,-h},
        {-h,-h,+h},{+h,-h,+h},{-h,+h,+h},{+h,+h,+h},
    };

    int minx=  1e9, miny=  1e9;
    int maxx= -1e9, maxy= -1e9;
    bool any=false;

    for (Vec3 p : corners){
        p = rotateZ_pt(p, P.rollZ);
        p = rotateX_pt(p, P.pitchX);
        Vec3 w = p + P.center;

        int px, py;
        if (project_from_camera(cb, w, width, height, px, py)) {
            any = true;
            minx = std::min(minx, px); miny = std::min(miny, py);
            maxx = std::max(maxx, px); maxy = std::max(maxy, py);
        }
    }

    Box2D b; b.label = P.label; b.valid = any && minx<=maxx && miny<=maxy;
    if (b.valid){
        b.x0 = std::clamp(minx, 0, width -1);
        b.y0 = std::clamp(miny, 0, height-1);
        b.x1 = std::clamp(maxx, 0, width -1);
        b.y1 = std::clamp(maxy, 0, height-1);
    }
    return b;
}

// Draw a 1px rectangle into an RGB framebuffer (row-major, top-left is (0,0) visually)
inline void draw_rect(std::vector<unsigned char>& fb, int width, int height,
                      int x0, int y0, int x1, int y1)
{
    x0 = std::clamp(x0,0,width-1); x1 = std::clamp(x1,0,width-1);
    y0 = std::clamp(y0,0,height-1); y1 = std::clamp(y1,0,height-1);
    if (x0>x1 || y0>y1) return;

    auto put = [&](int x,int y){
        int idx = (y * width + x) * 3;          // top-origin to match projection & framebuffer
        fb[idx+0]=255; fb[idx+1]=255; fb[idx+2]=0;
    };
    for (int x=x0; x<=x1; ++x){ put(x,y0); put(x,y1); }
    for (int y=y0; y<=y1; ++y){ put(x0,y); put(x1,y); }
}
