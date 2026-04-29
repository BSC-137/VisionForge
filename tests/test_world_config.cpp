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

    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "test_world_config: all tests passed\n";
    return 0;
}
