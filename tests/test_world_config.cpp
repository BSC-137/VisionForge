#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "visionforge/world_config.hpp"

namespace fs = std::filesystem;

static fs::path write_temp_json(const std::string& contents) {
    static uint64_t serial = 0;
    auto dir   = fs::temp_directory_path();
    auto path  = dir / ("vf_wcfg_test_" + std::to_string(++serial) + ".json");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("temp json write failed");
    out << contents;
    return path;
}

static bool contains_substr(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

#define REQUIRE_THROW_MSG(expr, needle)                                                              \
    do {                                                                                              \
        bool threw = false;                                                                             \
        std::string what;                                                                               \
        try {                                                                                         \
            expr;                                                                                     \
        } catch (const WorldConfigError& ex) {                                                         \
            threw = true;                                                                             \
            what  = ex.what();                                                                        \
        } catch (const std::exception& ex) {                                                           \
            threw = true;                                                                             \
            what  = ex.what();                                                                        \
        }                                                                                             \
        if (!threw) {                                                                                 \
            std::cerr << "FAIL: expected exception: " #expr "\n";                                      \
            return false;                                                                             \
        }                                                                                             \
        if (!contains_substr(what, needle)) {                                                         \
            std::cerr << "FAIL: wrong message for " #expr "\nGot: " << what << "\nExpected substring: " \
                      << needle << "\n";                                                               \
            return false;                                                                             \
        }                                                                                             \
    } while (0)

static bool test_canonical_minimal_strict_ok() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "dataset": {"root": "dataset"}
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = true});
        if (cfg.assets.size() != 1 || cfg.assets[0].path != "mesh.obj" || cfg.assets[0].name != "mesh") {
            std::cerr << "FAIL: canonical_minimal parse values\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: canonical_minimal threw: " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_unknown_root_key_strict() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj"}],
      "bogus_root_key": true
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "unknown key");
    return true;
}

static bool test_strict_rejects_albedo_path() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "m", "albedo_path": "x.jpg"}]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "deprecated key");
    return true;
}

static bool test_non_strict_maps_albedo_path() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "m", "albedo_path": "x.jpg"}]
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = false});
        if (cfg.assets[0].albedo_map != "x.jpg") {
            std::cerr << "FAIL: expected albedo_map mapped from albedo_path\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: non_strict alias: " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_missing_assets_array() {
    const std::string json = R"({
      "dataset": {"root": "dataset"}
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "must define");
    return true;
}

static bool test_scenario_missing_name() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{}]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "missing required field `name`");
    return true;
}

static bool test_unknown_asset_reference_in_scenario() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{
        "name": "s",
        "root_nodes": [{"name": "n", "asset": "does_not_exist"}]
      }]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "unknown asset");
    return true;
}

static bool test_strict_rejects_normal_path() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "m", "normal_path": "n.jpg"}]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "deprecated key");
    return true;
}

static bool test_strict_rejects_obj_field() {
    const std::string json = R"({
      "assets": [{"obj": "mesh.obj", "name": "m"}]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "deprecated key");
    return true;
}

static bool test_hdr_invalid_type_throws() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "m"}],
      "hdr": 42
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "must be a string path");
    return true;
}

static bool test_valid_scenario_asset_ref() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{
        "name": "s",
        "root_nodes": [{"name": "n", "asset": "mesh"}]
      }]
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = true});
        if (cfg.scenarios.size() != 1 || cfg.scenarios[0].root_nodes[0].asset != "mesh") {
            std::cerr << "FAIL: scenario parse\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: valid scenario: " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_trajectory_parse_and_interpolate() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{
        "name": "dolly",
        "camera": {
          "trajectory": [
            {"t": 0.0, "pos": [18.0, 8.0, 24.0]},
            {"t": 1.0, "pos": [-18.0, 8.0, 24.0]}
          ]
        }
      }]
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = true});
        if (cfg.scenarios.size() != 1) {
            std::cerr << "FAIL: trajectory_parse: expected 1 scenario\n";
            return false;
        }
        const auto& cam = cfg.scenarios[0].camera;
        if (cam.trajectory.size() != 2) {
            std::cerr << "FAIL: trajectory_parse: expected 2 keyframes, got " << cam.trajectory.size() << "\n";
            return false;
        }

        // alpha = 0.0 → keyframe 0 position
        Vec3 pos0 = interpolate_trajectory(cam, 0.0);
        if (std::abs(pos0.x - 18.0) > 1e-9 || std::abs(pos0.y - 8.0) > 1e-9 || std::abs(pos0.z - 24.0) > 1e-9) {
            std::cerr << "FAIL: trajectory alpha=0.0: expected (18,8,24), got ("
                      << pos0.x << "," << pos0.y << "," << pos0.z << ")\n";
            return false;
        }

        // alpha = 1.0 → keyframe 1 position
        Vec3 pos1 = interpolate_trajectory(cam, 1.0);
        if (std::abs(pos1.x - (-18.0)) > 1e-9 || std::abs(pos1.y - 8.0) > 1e-9 || std::abs(pos1.z - 24.0) > 1e-9) {
            std::cerr << "FAIL: trajectory alpha=1.0: expected (-18,8,24), got ("
                      << pos1.x << "," << pos1.y << "," << pos1.z << ")\n";
            return false;
        }

        // alpha = 0.5 → midpoint (0, 8, 24)
        Vec3 pos_mid = interpolate_trajectory(cam, 0.5);
        if (std::abs(pos_mid.x - 0.0) > 1e-9 || std::abs(pos_mid.y - 8.0) > 1e-9 || std::abs(pos_mid.z - 24.0) > 1e-9) {
            std::cerr << "FAIL: trajectory alpha=0.5: expected (0,8,24), got ("
                      << pos_mid.x << "," << pos_mid.y << "," << pos_mid.z << ")\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: trajectory_parse_and_interpolate: " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_trajectory_empty_falls_back_to_lookfrom_max() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{
        "name": "static",
        "camera": {
          "lookfrom": [10.0, 5.0, 15.0]
        }
      }]
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = true});
        const auto& cam = cfg.scenarios[0].camera;
        if (!cam.trajectory.empty()) {
            std::cerr << "FAIL: trajectory should be empty when not specified\n";
            return false;
        }
        Vec3 lf = interpolate_trajectory(cam, 0.5);
        if (std::abs(lf.x - 10.0) > 1e-9 || std::abs(lf.y - 5.0) > 1e-9 || std::abs(lf.z - 15.0) > 1e-9) {
            std::cerr << "FAIL: fallback to lookfrom.max: expected (10,5,15), got ("
                      << lf.x << "," << lf.y << "," << lf.z << ")\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: trajectory_empty_fallback: " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_trajectory_unknown_key_strict_rejects() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "scenarios": [{
        "name": "s",
        "camera": {
          "trajectory": [
            {"t": 0.0, "pos": [0, 0, 0], "bogus": 1}
          ]
        }
      }]
    })";
    REQUIRE_THROW_MSG(load_world_config(write_temp_json(json).string(), {.strict = true}), "unknown key");
    return true;
}

int main() {
    int failures = 0;
    if (!test_canonical_minimal_strict_ok()) ++failures;
    if (!test_unknown_root_key_strict()) ++failures;
    if (!test_strict_rejects_albedo_path()) ++failures;
    if (!test_non_strict_maps_albedo_path()) ++failures;
    if (!test_missing_assets_array()) ++failures;
    if (!test_scenario_missing_name()) ++failures;
    if (!test_unknown_asset_reference_in_scenario()) ++failures;
    if (!test_valid_scenario_asset_ref()) ++failures;
    if (!test_strict_rejects_normal_path()) ++failures;
    if (!test_strict_rejects_obj_field()) ++failures;
    if (!test_hdr_invalid_type_throws()) ++failures;
    if (!test_trajectory_parse_and_interpolate()) ++failures;
    if (!test_trajectory_empty_falls_back_to_lookfrom_max()) ++failures;
    if (!test_trajectory_unknown_key_strict_rejects()) ++failures;

    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "test_world_config: all tests passed\n";
    return 0;
}
