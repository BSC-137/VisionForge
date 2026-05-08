// Acceptance checks for meta_pose.hpp (camera c2w + intrinsics + ray consistency).
#include <cmath>
#include <cstdio>
#include <iostream>

#include "visionforge/camera.hpp"
#include "visionforge/meta_pose.hpp"
#include "visionforge/ray.hpp"
#include "visionforge/vec3.hpp"

static int failures = 0;

static void expect_near(const char* what, double a, double b, double eps) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "FAIL " << what << " got " << a << " expected " << b << "\n";
        ++failures;
    }
}

static void test_intrinsics_roundtrip() {
    const int W = 65, H = 33;
    const double vfov = 38.0;
    double fx, fy, cx, cy;
    vf::meta_pose::pinhole_intrinsics_match_renderer(W, H, vfov, fx, fy, cx, cy);
    const double aspect = double(W) / double(H);
    const double vh = 2.0 * std::tan(0.5 * vfov * (PI / 180.0));
    const double vw = aspect * vh;
    expect_near("fx", fx, double(W - 1) / vw, 1e-9);
    expect_near("fy", fy, double(H - 1) / vh, 1e-9);
    expect_near("cx", cx, 0.5 * double(W - 1), 1e-9);
    expect_near("cy", cy, 0.5 * double(H - 1), 1e-9);
}

static void test_c2w_orthonormal() {
    Camera cam(Vec3(3.1, 2.0, 5.5), Vec3(0, 1, 0), Vec3(0, 1, 0), 42.0, 16.0 / 9.0, 0.0, 4.0, 0.0, 1.0);
    double M[16];
    vf::meta_pose::camera_c2w_row_major_opencv(cam, M);
    // R is rows 0..2, cols 0..2 in row-major top-left
    double RRT[9] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k)
                s += M[i * 4 + k] * M[j * 4 + k]; // R[i,k]*R[j,k] => (R*R^T)[i,j]
            RRT[i * 3 + j] = s;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "R*R^T[%d,%d]", i, j);
            expect_near(buf, RRT[i * 3 + j], want, 1e-10);
        }
    }
    expect_near("translation x", M[3], cam.origin.x, 1e-12);
    expect_near("translation y", M[7], cam.origin.y, 1e-12);
    expect_near("translation z", M[11], cam.origin.z, 1e-12);
}

static Vec3 ray_dir_world(const Camera& cam, double s, double t) {
    const Ray r = cam.get_ray(s, t);
    return normalize(r.direction);
}

int main() {
    test_intrinsics_roundtrip();
    test_c2w_orthonormal();

    const int W = 64, H = 36;
    const double vfov = 35.0;
    const double aspect = double(W) / double(H);
    Camera cam(Vec3(11, 6, 18), Vec3(0, 1, 0), Vec3(0, 1, 0), vfov, aspect, 0.0,
               (Vec3(11, 6, 18) - Vec3(0, 1, 0)).length(), 0.0, 1.0);
    double fx, fy, cx, cy;
    vf::meta_pose::pinhole_intrinsics_match_renderer(W, H, vfov, fx, fy, cx, cy);
    double c2w[16];
    vf::meta_pose::camera_c2w_row_major_opencv(cam, c2w);

    for (double s : {0.12, 0.5, 0.88}) {
        for (double t : {0.11, 0.52, 0.91}) {
            const Vec3 d_engine = ray_dir_world(cam, s, t);
            const double i = s * double(W - 1);
            const double j = t * double(H - 1);
            const double dcx = (i - cx) / fx;
            const double dcy = -(j - cy) / fy;
            const double dcz = 1.0;
            const double wx = c2w[0] * dcx + c2w[1] * dcy + c2w[2] * dcz;
            const double wy = c2w[4] * dcx + c2w[5] * dcy + c2w[6] * dcz;
            const double wz = c2w[8] * dcx + c2w[9] * dcy + c2w[10] * dcz;
            Vec3 d_recon = normalize(Vec3(wx, wy, wz));
            const double d_dot = dot(d_engine, d_recon);
            if (std::fabs(d_dot - 1.0) > 2e-5) {
                std::cerr << "FAIL ray match s=" << s << " t=" << t << " dot=" << d_dot << "\n";
                ++failures;
            }
        }
    }

    if (failures != 0) {
        std::cerr << "test_meta_pose: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test_meta_pose: ok\n";
    return 0;
}
