#include "visionforge/dataset_manifest.hpp"

#include "visionforge/config_keys.hpp"
#include "visionforge/world_config.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(VF_USE_OMP) && defined(_OPENMP)
#include <omp.h>
#endif

namespace vf {
namespace {

namespace ck = visionforge::cfg;
using json = nlohmann::json;

json vec3_json(const Vec3& v) { return json::array({v.x, v.y, v.z}); }

json scalar_range_json(const ScalarRange& r) {
    if (r.min == r.max) return r.min;
    return json{{std::string(ck::k_min), r.min}, {std::string(ck::k_max), r.max}};
}

json vec3_range_json(const Vec3Range& r) {
    const bool degenerate = (r.min.x == r.max.x && r.min.y == r.max.y && r.min.z == r.max.z);
    if (degenerate) return vec3_json(r.min);
    return json{{std::string(ck::k_min), vec3_json(r.min)}, {std::string(ck::k_max), vec3_json(r.max)}};
}

json asset_entry_json(const AssetEntry& e) {
    json a = {
        {std::string(ck::k_name), e.name},
        {std::string(ck::k_path), e.path},
        {std::string(ck::k_weight), e.weight},
        {std::string(ck::k_scale), e.scale},
        {std::string(ck::k_color), e.color},
        {std::string(ck::k_label), e.label},
        {std::string(ck::k_class_id), e.class_id},
        {std::string(ck::k_y_offset), e.y_offset},
        {std::string(ck::k_roughness), scalar_range_json(e.roughness)},
        {std::string(ck::k_metallic), scalar_range_json(e.metallic)},
    };
    if (!e.albedo_map.empty()) a[std::string(ck::k_albedo_map)] = e.albedo_map;
    if (!e.normal_map.empty()) a[std::string(ck::k_normal_map)] = e.normal_map;
    if (!e.roughness_map.empty()) a[std::string(ck::k_roughness_map)] = e.roughness_map;
    if (!e.metallic_map.empty()) a[std::string(ck::k_metallic_map)] = e.metallic_map;
    return a;
}

json hdr_json(const WorldConfig& cfg) {
    return json{{std::string(ck::k_path), cfg.hdr_path}, {std::string(ck::k_intensity), cfg.hdr_intensity}};
}

json node_tree_json(const WorldConfig::NodeEntry& node);

json node_tree_json(const WorldConfig::NodeEntry& node) {
    json j = {
        {std::string(ck::k_name), node.name},
        {std::string(ck::k_asset), node.asset},
        {std::string(ck::k_position), vec3_json(node.position)},
        {std::string(ck::k_rotation), vec3_json(node.rotation)},
        {std::string(ck::k_scale), vec3_json(node.scale)},
        {std::string(ck::k_grounding_constraint), node.grounding_constraint},
        {std::string(ck::k_y_offset), node.y_offset},
    };
    json ch = json::array();
    for (const auto& c : node.children) ch.push_back(node_tree_json(c));
    j[std::string(ck::k_children)] = std::move(ch);
    return j;
}

json scenario_json(const WorldConfig::Scenario& s) {
    json cam = {
        {std::string(ck::k_lookat), vec3_json(s.camera.lookat)},
        {std::string(ck::k_up), vec3_json(s.camera.up)},
        {std::string(ck::k_lookfrom), vec3_range_json(s.camera.lookfrom)},
        {std::string(ck::k_fov_deg), scalar_range_json(s.camera.fov_deg)},
    };
    json roots = json::array();
    for (const auto& n : s.root_nodes) roots.push_back(node_tree_json(n));
    return json{{std::string(ck::k_name), s.name}, {std::string(ck::k_camera), std::move(cam)},
                {std::string(ck::k_root_nodes), std::move(roots)}};
}

std::string platform_string() {
#if defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "darwin";
#else
    return "unknown";
#endif
}

std::optional<std::string> cpu_model_best_effort() {
#if defined(__linux__)
    std::ifstream in("/proc/cpuinfo");
    if (!in) return std::nullopt;
    std::string line;
    while (std::getline(in, line)) {
        constexpr std::string_view prefix = "model name\t: ";
        if (line.compare(0, prefix.size(), prefix.data()) == 0)
            return line.substr(prefix.size());
        constexpr std::string_view prefix2 = "model name : ";
        if (line.size() > prefix2.size() && line.compare(0, prefix2.size(), prefix2.data()) == 0)
            return line.substr(prefix2.size());
    }
#endif
    return std::nullopt;
}

json system_json() {
    json s = {{"platform", platform_string()}};
    auto cpu = cpu_model_best_effort();
    if (cpu && !cpu->empty()) s["cpu_model"] = *cpu;

#if defined(VF_USE_OMP) && defined(_OPENMP)
    const int omp_max = omp_get_max_threads();
    s["omp_max_threads"] = omp_max;
    s["threads_used"] = omp_max;
    s["parallel_notes"] =
        "threads_used matches omp_get_max_threads() (expected team size for parallel loops; not sampled inside "
        "regions).";
#else
    s["omp_max_threads"] = 1;
    s["threads_used"] = 1;
    s["parallel_notes"] = "Built without OpenMP (VF_USE_OMP undefined or OpenMP not linked).";
#endif
    return s;
}

json random_seed_json(const DatasetManifestParams& p) {
    json seed_doc = {
        {"render_config_seed", p.render_seed},
        {"thread_rng_per_row_legacy_preview",
         "Inside some preview/lighting-only paths each scanline calls vf_rng::seed_thread_rng(row_mix + constant); "
         "forge/scenario dataset frames use seed_thread_rng(uint64_t(render_config_seed) + frame_index)."},
        {"random_double_source", "vf_rng thread-local xoshiro256+ (see vec3.hpp)."},
    };
    if (!p.dataset_rng_note.empty()) seed_doc["dataset_domain_randomization"] = p.dataset_rng_note;
    return seed_doc;
}

json dataset_layout_json(const DatasetManifestParams& p) {
    json d = {{"job_kind", p.job_kind},
              {"dataset_root", p.dataset_root},
              {"frames_total", p.frames_total},
              {"train_frames", p.train_frames},
              {"val_frames", p.val_frames},
              {"train_subdirectory", "train"},
              {"val_subdirectory", "val"},
              {"forge_io_had_error", p.forge_io_had_error}};
    if (p.active_scenario) d["active_scenario"] = *p.active_scenario;

    if (p.job_kind == "forge") {
        d["rgb_stem_pattern"] = "frame_%04d";
        d["sidecar_suffixes"] = json::array({".png", "_spatial.exr", "_meta.json", ".txt"});
        d["coco_annotations_file"] = "annotations_coco.json";
    } else if (p.job_kind == "scenario") {
        d["rgb_stem_pattern"] = "sfrm_%04d";
        d["sidecar_suffixes"] = json::array({".png", "_spatial.exr", "_meta.json", ".txt"});
        d["coco_annotations_file"] = "scenario_coco.json";
    }
    return d;
}

#ifndef VF_ENGINE_VERSION_STR
#define VF_ENGINE_VERSION_STR "unknown"
#endif

#ifndef VF_GIT_COMMIT_STR
#define VF_GIT_COMMIT_STR "unknown"
#endif

json version_info_json() {
    json v = {{"engine_version", VF_ENGINE_VERSION_STR},
              {"schema_version", k_manifest_schema_version},
              {"git_commit", VF_GIT_COMMIT_STR}};
#if defined(VF_GIT_DIRTY)
    v["git_dirty"] = true;
#endif
    return v;
}

std::string dataset_id_hex(const DatasetManifestParams& p) {
    std::string payload = p.dataset_root;
    payload.push_back('\n');
    payload += p.timestamp_iso_utc;
    payload.push_back('\n');
    payload += p.argv_joined;
    payload.push_back('\n');
    payload += std::to_string(p.render_seed);
    payload.push_back('\n');
    payload += p.job_kind;
    const std::string full = sha256_hex(payload);
    return full.size() >= 32 ? full.substr(0, 32) : full;
}

// ----- SHA-256 (RFC 6234 style), lowercase hex -----
constexpr std::array<uint32_t, 64> k_sha256_k = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + S1 + ch + k_sha256_k[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t0 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t0 + t1;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace

nlohmann::json world_config_to_resolved_json(const WorldConfig& cfg) {
    json assets = json::array();
    for (const auto& a : cfg.assets) assets.push_back(asset_entry_json(a));

    json scenarios = json::array();
    for (const auto& s : cfg.scenarios) scenarios.push_back(scenario_json(s));

    json root = {
        {std::string(ck::k_render),
         json{{std::string(ck::k_width), cfg.render.width},
              {std::string(ck::k_height), cfg.render.height},
              {std::string(ck::k_spp), cfg.render.spp},
              {std::string(ck::k_max_depth), cfg.render.max_depth},
              {std::string(ck::k_exposure), cfg.render.exposure},
              {std::string(ck::k_sky_gain), cfg.render.sky_gain},
              {std::string(ck::k_preview), cfg.render.preview},
              {std::string(ck::k_seed), cfg.render.seed}}},
        {std::string(ck::k_camera),
         json{{std::string(ck::k_lookat), vec3_json(cfg.camera.lookat)},
              {std::string(ck::k_up), vec3_json(cfg.camera.up)},
              {std::string(ck::k_lookfrom), vec3_range_json(cfg.camera.lookfrom)},
              {std::string(ck::k_fov_deg), scalar_range_json(cfg.camera.fov_deg)}}},
        {std::string(ck::k_lighting),
         json{{std::string(ck::k_sun_azimuth_deg), scalar_range_json(cfg.lighting.sun_azimuth_deg)},
              {std::string(ck::k_sun_elevation_deg), scalar_range_json(cfg.lighting.sun_elevation_deg)}}},
        {std::string(ck::k_terrain),
         json{{std::string(ck::k_bounds),
               json{{std::string(ck::k_xmin), cfg.terrain.xmin},
                    {std::string(ck::k_xmax), cfg.terrain.xmax},
                    {std::string(ck::k_zmin), cfg.terrain.zmin},
                    {std::string(ck::k_zmax), cfg.terrain.zmax}}},
              {std::string(ck::k_amp), cfg.terrain.amp},
              {std::string(ck::k_scale), cfg.terrain.scale},
              {std::string(ck::k_nx), cfg.terrain.nx},
              {std::string(ck::k_nz), cfg.terrain.nz}}},
        {std::string(ck::k_assets), std::move(assets)},
        {"hdr", hdr_json(cfg)},
        {"ground_tex", cfg.ground_tex},
        {"ground_scale", cfg.ground_scale},
        {std::string(ck::k_placement),
         json{{std::string(ck::k_x), scalar_range_json(cfg.placement.x)},
              {std::string(ck::k_z), scalar_range_json(cfg.placement.z)},
              {std::string(ck::k_yaw_deg), scalar_range_json(cfg.placement.yaw_deg)}}},
        {std::string(ck::k_dataset),
         json{{std::string(ck::k_root), cfg.dataset.root}, {std::string(ck::k_train_split), cfg.dataset.train_split}}},
        {std::string(ck::k_scenarios), std::move(scenarios)},
    };

    // Mirror legacy single-asset mirror used internally after parsing.
    if (!cfg.assets.empty()) {
        const auto& first = cfg.assets[0];
        root[std::string(ck::k_asset)] =
            json{{std::string(ck::k_path), first.path},
                 {std::string(ck::k_scale), first.scale},
                 {std::string(ck::k_color), first.color},
                 {std::string(ck::k_label), first.label},
                 {std::string(ck::k_class_id), first.class_id},
                 {std::string(ck::k_y_offset), first.y_offset},
                 {std::string(ck::k_roughness), scalar_range_json(first.roughness)},
                 {std::string(ck::k_metallic), scalar_range_json(first.metallic)}};
    }

    return root;
}

std::string sha256_hex(std::string_view data) {
    uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t bit_len = uint64_t(data.size()) * 8;
    std::vector<uint8_t> buf(data.begin(), data.end());
    buf.push_back(0x80);
    while ((buf.size() % 64) != 56) buf.push_back(0);
    for (int i = 7; i >= 0; --i) buf.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xff));

    for (size_t off = 0; off < buf.size(); off += 64) sha256_compress(state, buf.data() + off);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned i = 0; i < 8; ++i) oss << std::setw(8) << state[i];
    std::string out = oss.str();
    for (char& ch : out) {
        if (ch >= 'A' && ch <= 'F') ch = char(ch - 'A' + 'a');
    }
    return out;
}

bool write_dataset_manifest_atomic(const std::filesystem::path& dest_path,
                                   const DatasetManifestParams& params) {
    json resolved = params.resolved_config;
    std::optional<std::string> cfg_sha;
    try {
        const std::string dumped = resolved.dump();
        cfg_sha = sha256_hex(dumped);
    } catch (...) {
        cfg_sha = std::nullopt;
    }

    json config_audit = {
        {"config_path", params.config_path},
        {"resolved_config", std::move(resolved)},
    };
    if (cfg_sha) config_audit["resolved_config_sha256"] = *cfg_sha;

    json exec = {{"argv", params.argv_joined},
                 {"cwd", params.cwd},
                 {"timestamp_utc", params.timestamp_iso_utc},
                 {"random_seed", random_seed_json(params)},
                 {"dataset_id", dataset_id_hex(params)}};
    if (params.active_scenario) exec["active_scenario"] = *params.active_scenario;

    json doc = {{"version_info", version_info_json()},
                {"execution", std::move(exec)},
                {"config_audit", std::move(config_audit)},
                {"system", system_json()},
                {"dataset_layout", dataset_layout_json(params)}};

    if (!params.render_summary.is_null() && !params.render_summary.empty())
        doc["render_summary"] = params.render_summary;

    std::filesystem::path tmp = dest_path;
    tmp += ".tmp";

    try {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << doc.dump(2);
        out.flush();
        if (!out) {
            std::cerr << "[manifest] ERROR: failed writing temporary manifest file: " << tmp.string() << "\n";
            std::filesystem::remove(tmp);
            return false;
        }
        out.close();

        std::error_code ec;
        std::filesystem::rename(tmp, dest_path, ec);
        if (ec) {
            std::cerr << "[manifest] ERROR: atomic rename failed for manifest.json (" << ec.message()
                      << "). Temporary path: " << tmp.string() << "\n";
            std::filesystem::remove(tmp);
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[manifest] ERROR: exception while writing manifest.json: " << ex.what() << "\n";
        std::error_code ignore;
        std::filesystem::remove(tmp, ignore);
        return false;
    }

    return true;
}

} // namespace vf
