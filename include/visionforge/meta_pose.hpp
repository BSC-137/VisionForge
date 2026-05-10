#pragma once

#include <cmath>
#include "visionforge/camera.hpp"
#include "visionforge/vec3.hpp"

namespace vf {
namespace meta_pose {

namespace detail {
inline void mat3_mul_rm(const double A[9], const double B[9], double C[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] + A[i * 3 + 2] * B[2 * 3 + j];
        }
    }
}
inline void mat3_Rx(double c, double s, double R[9]) {
    R[0] = 1;
    R[1] = 0;
    R[2] = 0;
    R[3] = 0;
    R[4] = c;
    R[5] = -s;
    R[6] = 0;
    R[7] = s;
    R[8] = c;
}
inline void mat3_Ry(double c, double s, double R[9]) {
    R[0] = c;
    R[1] = 0;
    R[2] = s;
    R[3] = 0;
    R[4] = 1;
    R[5] = 0;
    R[6] = -s;
    R[7] = 0;
    R[8] = c;
}
inline void mat3_Rz(double c, double s, double R[9]) {
    R[0] = c;
    R[1] = -s;
    R[2] = 0;
    R[3] = s;
    R[4] = c;
    R[5] = 0;
    R[6] = 0;
    R[7] = 0;
    R[8] = 1;
}
} // namespace detail

// Pinhole intrinsics matching VisionForge Camera + main render loops:
// pixel i maps to s = i / max(W-1,1), j to t = j / max(H-1,1) in get_ray(s,t).
// vw = 2*tan(vfov/2)*(W/H), vh = 2*tan(vfov/2); forward ray (unnormalized) in "camera" coords
// proportional to ((s-0.5)*vw, -(t-0.5)*vh, 1) before applying rotation to world.
inline void pinhole_intrinsics_match_renderer(int W, int H, double vfov_deg, double& fx, double& fy, double& cx,
                                              double& cy) {
    const double aspect = (H > 0) ? (double(W) / double(H)) : 1.0;
    const double vfov = vfov_deg * (PI / 180.0);
    const double vh = 2.0 * std::tan(0.5 * vfov);
    const double vw = aspect * vh;
    const double wm = W > 1 ? double(W - 1) : 1.0;
    const double hm = H > 1 ? double(H - 1) : 1.0;
    fx = wm / vw;
    fy = hm / vh;
    cx = 0.5 * wm;
    cy = 0.5 * hm;
}

// Columns of rotation: u, -v, -w; row-major 4x4; P_world = M * P_cam (homogeneous column).
inline void camera_c2w_from_basis_row_major(const Vec3& u, const Vec3& v, const Vec3& w, const Vec3& origin,
                                            double m16[16]) {
    m16[0] = u.x;
    m16[1] = -v.x;
    m16[2] = -w.x;
    m16[3] = origin.x;
    m16[4] = u.y;
    m16[5] = -v.y;
    m16[6] = -w.y;
    m16[7] = origin.y;
    m16[8] = u.z;
    m16[9] = -v.z;
    m16[10] = -w.z;
    m16[11] = origin.z;
    m16[12] = 0.0;
    m16[13] = 0.0;
    m16[14] = 0.0;
    m16[15] = 1.0;
}

// OpenCV-style right-handed camera: +X right (u), +Y down (-VForge v when t increases),
// +Z forward into scene (-w). 4x4 is row-major; P_world = M * P_cam (homogeneous column).
inline void camera_c2w_row_major_opencv(const Camera& cam, double m16[16]) {
    camera_c2w_from_basis_row_major(cam.u, cam.v, cam.w, cam.origin, m16);
}

// Logical object pose: v_world = R * diag(scale) * v_local + position with R = Rz * Ry * Rx (column-vector
// convention), matching SceneNode::flat_pack: rot_x, then rot_y, then rot_z on the scaled vertex.
// Omits slope_alignment and per-frame vertex snap applied inside flat_pack when grounding_constraint is true
// (see JSON field transform_supervision on each object).
inline void logical_object_c2w_row_major(const Vec3& rot_deg, const Vec3& scale, const Vec3& position,
                                         double m16[16]) {
    const double rx = rot_deg.x * (PI / 180.0);
    const double ry = rot_deg.y * (PI / 180.0);
    const double rz = rot_deg.z * (PI / 180.0);
    double Rx[9], Ry[9], Rz[9], Rtmp[9], R[9];
    detail::mat3_Rx(std::cos(rx), std::sin(rx), Rx);
    detail::mat3_Ry(std::cos(ry), std::sin(ry), Ry);
    detail::mat3_Rz(std::cos(rz), std::sin(rz), Rz);
    detail::mat3_mul_rm(Ry, Rx, Rtmp);
    detail::mat3_mul_rm(Rz, Rtmp, R);

    const double sx_ = scale.x, sy_ = scale.y, sz_ = scale.z;
    m16[0] = R[0] * sx_;
    m16[1] = R[1] * sy_;
    m16[2] = R[2] * sz_;
    m16[3] = position.x;
    m16[4] = R[3] * sx_;
    m16[5] = R[4] * sy_;
    m16[6] = R[5] * sz_;
    m16[7] = position.y;
    m16[8] = R[6] * sx_;
    m16[9] = R[7] * sy_;
    m16[10] = R[8] * sz_;
    m16[11] = position.z;
    m16[12] = 0.0;
    m16[13] = 0.0;
    m16[14] = 0.0;
    m16[15] = 1.0;
}

} // namespace meta_pose
} // namespace vf
