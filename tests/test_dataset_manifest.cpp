#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "visionforge/dataset_manifest.hpp"
#include "visionforge/world_config.hpp"

namespace fs = std::filesystem;

static fs::path write_temp_json(const std::string& contents) {
    static uint64_t serial = 0;
    auto dir  = fs::temp_directory_path();
    auto path = dir / ("vf_manifest_test_" + std::to_string(++serial) + ".json");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("temp json write failed");
    out << contents;
    return path;
}

static bool test_world_config_roundtrip_shape() {
    const std::string json = R"({
      "assets": [{"path": "mesh.obj", "name": "mesh"}],
      "dataset": {"root": "dataset"}
    })";
    try {
        WorldConfig cfg = load_world_config(write_temp_json(json).string(), {.strict = true});
        nlohmann::json j = vf::world_config_to_resolved_json(cfg);
        if (!j.contains("render") || !j.contains("assets")) {
            std::cerr << "FAIL: resolved_config missing keys\n";
            return false;
        }
        if (!j["assets"].is_array() || j["assets"].empty()) {
            std::cerr << "FAIL: assets array\n";
            return false;
        }
        if (j["assets"][0]["path"] != "mesh.obj") {
            std::cerr << "FAIL: asset path canonicalization\n";
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: exception " << ex.what() << "\n";
        return false;
    }
    return true;
}

static bool test_sha256_known_vector() {
    // RFC 6234 / NIST vector for "abc"
    const std::string got = vf::sha256_hex("abc");
    const char* expected =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    if (got != expected) {
        std::cerr << "FAIL: sha256 mismatch\nGot: " << got << "\n";
        return false;
    }
    return true;
}

static bool test_manifest_write_atomic() {
    static uint64_t serial = 0;
    auto dir  = fs::temp_directory_path();
    auto root = dir / ("vf_manifest_atomic_" + std::to_string(++serial));
    fs::create_directories(root);
    const fs::path dest = root / "manifest.json";

    vf::DatasetManifestParams p;
    p.argv_joined           = "visionforge forge --config x.json --frames 1";
    p.cwd                   = "/tmp";
    p.timestamp_iso_utc     = "2026-05-07T12:00:00Z";
    p.job_kind              = "forge";
    p.config_path           = "x.json";
    p.resolved_config       = nlohmann::json::object({{"ok", true}});
    p.frames_total          = 10;
    p.train_frames          = 8;
    p.val_frames            = 2;
    p.dataset_root          = root.string();
    p.render_seed           = 42;
    p.dataset_rng_note      = "test note";

    if (!vf::write_dataset_manifest_atomic(dest, p)) {
        std::cerr << "FAIL: atomic manifest write\n";
        return false;
    }
    std::ifstream in(dest);
    nlohmann::json doc;
    in >> doc;
    if (!doc.contains("version_info") || doc["version_info"]["schema_version"] != vf::k_manifest_schema_version) {
        std::cerr << "FAIL: manifest schema_version\n";
        return false;
    }
    if (!doc.contains("execution") || doc["execution"]["dataset_id"].get<std::string>().size() != 32) {
        std::cerr << "FAIL: dataset_id\n";
        return false;
    }
    return true;
}

int main() {
    int failures = 0;
    if (!test_world_config_roundtrip_shape()) ++failures;
    if (!test_sha256_known_vector()) ++failures;
    if (!test_manifest_write_atomic()) ++failures;

    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "test_dataset_manifest: all tests passed\n";
    return 0;
}
