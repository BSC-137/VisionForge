// Tests for Camera::world_to_camera() and the optical-flow projection math.
//
// Two cameras are constructed:
//   curr_cam  at (0, 0, 10) looking at the origin
//   prev_cam  at (1, 0, 10) looking at the origin
//
// A world point at the origin (0,0,0) is projected through both cameras.
// The current camera sees the point at its image centre; the previous camera
// is shifted 1 unit to the right, so the projected column differs (non-zero
// flow_x), while the row difference should be near zero.

#include <cassert>
#include <cmath>
#include <cstdio>

#include "visionforge/camera.hpp"
#include "visionforge/meta_pose.hpp"

static bool approx_eq(double a, double b, double tol = 1e-4) {
    return std::fabs(a - b) <= tol;
}

// --- helpers shared with the flow post-pass in main.cpp ---
static void project_world_point(const Camera& cam, const Vec3& world_pt,
                                 double fx, double fy, double cx, double cy,
                                 double& u_out, double& v_out, bool& in_front) {
    Vec3 p_cam = cam.world_to_camera(world_pt);
    in_front = (p_cam.z > 0.0);
    if (!in_front) { u_out = v_out = 0.0; return; }
    u_out = fx * p_cam.x / p_cam.z + cx;
    v_out = fy * p_cam.y / p_cam.z + cy;
}

static bool test_world_to_camera_centre_ray() {
    // Camera at (0,0,10) looking at (0,0,0), FOV = 45 deg.
    Vec3 from(0, 0, 10), at(0, 0, 0), up(0, 1, 0);
    Camera cam(from, at, up, 45.0, 1.0, 0.0, 10.0, 0.0, 1.0);

    // A point exactly on the optical axis.
    Vec3 pt(0, 0, 0);
    Vec3 p_cam = cam.world_to_camera(pt);

    // x and y components must be 0 (on axis), z must be positive (in front).
    if (!approx_eq(p_cam.x, 0.0, 1e-9)) {
        std::fprintf(stderr, "FAIL: centre ray p_cam.x != 0, got %.10g\n", p_cam.x);
        return false;
    }
    if (!approx_eq(p_cam.y, 0.0, 1e-9)) {
        std::fprintf(stderr, "FAIL: centre ray p_cam.y != 0, got %.10g\n", p_cam.y);
        return false;
    }
    if (p_cam.z <= 0.0) {
        std::fprintf(stderr, "FAIL: centre ray p_cam.z <= 0, got %.10g\n", p_cam.z);
        return false;
    }
    return true;
}

static bool test_flow_x_nonzero_flow_y_near_zero() {
    // Both cameras look at the origin but are offset in X by 1 unit.
    // We use a world point at (0, 0, 5) — midway between origin and curr_cam.
    // This point lies exactly on curr_cam's optical axis (x=0, y=0 in that camera)
    // but is off-axis for prev_cam (shifted 1 unit right), so flow_x must be non-zero.
    const int W = 64, H = 64;
    const double vfov = 45.0;
    const double aspect = double(W) / double(H);  // 1.0

    Vec3 lookat(0, 0, 0), up(0, 1, 0);

    Vec3 curr_pos(0, 0, 10);
    double fd_curr = (curr_pos - lookat).length();
    Camera curr_cam(curr_pos, lookat, up, vfov, aspect, 0.0, fd_curr, 0.0, 1.0);

    Vec3 prev_pos(1, 0, 10);
    double fd_prev = (prev_pos - lookat).length();
    Camera prev_cam(prev_pos, lookat, up, vfov, aspect, 0.0, fd_prev, 0.0, 1.0);

    double fx, fy, cx, cy;
    vf::meta_pose::pinhole_intrinsics_match_renderer(W, H, vfov, fx, fy, cx, cy);

    // Point on curr_cam's optical axis (x=0, y=0 in camera space).
    // NOTE: the world origin (0,0,0) is the shared lookat of both cameras, so it projects
    // to the image centre in BOTH cameras — zero flow.  Use (0,0,5) instead, which sits on
    // curr_cam's axis but is off-axis in prev_cam's frame.
    Vec3 world_pt(0, 0, 5);

    // Current camera: point should project to image centre.
    double u_curr, v_curr;
    bool in_front;
    project_world_point(curr_cam, world_pt, fx, fy, cx, cy, u_curr, v_curr, in_front);
    if (!in_front) {
        std::fprintf(stderr, "FAIL: world_pt (0,0,5) is behind curr_cam\n");
        return false;
    }
    if (!approx_eq(u_curr, cx, 1.0)) {
        std::fprintf(stderr, "FAIL: u_curr=%.4g expected ~cx=%.4g\n", u_curr, cx);
        return false;
    }
    if (!approx_eq(v_curr, cy, 1.0)) {
        std::fprintf(stderr, "FAIL: v_curr=%.4g expected ~cy=%.4g\n", v_curr, cy);
        return false;
    }

    // Previous camera (shifted +1 in X): point should project to a different column.
    double u_prev, v_prev;
    project_world_point(prev_cam, world_pt, fx, fy, cx, cy, u_prev, v_prev, in_front);
    if (!in_front) {
        std::fprintf(stderr, "FAIL: world_pt (0,0,5) is behind prev_cam\n");
        return false;
    }

    // flow_x = u_prev - u_curr. The prev_cam is shifted +1 in X, so the world point
    // appears at a negative x offset in prev_cam space → u_prev < cx → flow_x < 0.
    const double flow_x = u_prev - u_curr;
    const double flow_y = v_prev - v_curr;

    if (std::fabs(flow_x) < 0.5) {
        std::fprintf(stderr,
                     "FAIL: flow_x=%.6g should be non-zero (prev_cam shifted 1 unit in X)\n"
                     "      u_curr=%.4g  u_prev=%.4g  cx=%.4g\n",
                     flow_x, u_curr, u_prev, cx);
        return false;
    }
    if (std::fabs(flow_y) > 0.5) {
        std::fprintf(stderr, "FAIL: flow_y=%.6g should be near zero (cameras at same Y)\n", flow_y);
        return false;
    }

    return true;
}

static bool test_point_behind_camera_returns_no_flow() {
    Vec3 from(0, 0, 10), at(0, 0, 0), up(0, 1, 0);
    Camera cam(from, at, up, 45.0, 1.0, 0.0, 10.0, 0.0, 1.0);

    // A point directly behind the camera.
    Vec3 behind(0, 0, 20);
    Vec3 p_cam = cam.world_to_camera(behind);

    // z should be negative (behind camera).
    if (p_cam.z >= 0.0) {
        std::fprintf(stderr, "FAIL: point behind camera should have p_cam.z < 0, got %.10g\n", p_cam.z);
        return false;
    }
    return true;
}

int main() {
    int failures = 0;
    if (!test_world_to_camera_centre_ray())        ++failures;
    if (!test_flow_x_nonzero_flow_y_near_zero())   ++failures;
    if (!test_point_behind_camera_returns_no_flow()) ++failures;

    if (failures) {
        std::fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("test_flow_math: all tests passed\n");
    return 0;
}
