#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

struct WorldConfig;

namespace vf {

/// Bump when `manifest.json` shape changes (keep README in sync).
inline constexpr const char k_manifest_schema_version[] = "2";

struct DatasetManifestParams {
    std::string argv_joined;
    std::string cwd;

    /// UTC ISO-8601 timestamp string (caller supplies for consistent dataset_id hashing).
    std::string timestamp_iso_utc;

    /// One of: forge, scenario, legacy_cli
    std::string job_kind;

    std::string config_path;
    nlohmann::json resolved_config;

    std::optional<std::string> active_scenario;

    int frames_total = 0;
    int train_frames = 0;
    int val_frames = 0;

    /// Absolute or normalized dataset root (directory containing manifest.json).
    std::string dataset_root;

    unsigned render_seed = 0;

    /// Human-readable notes on non-render RNG streams (placement DR, etc.).
    std::string dataset_rng_note;

    /// Optional render/time summary (legacy single-shot or aggregate forge timings).
    nlohmann::json render_summary;

    bool forge_io_had_error = false;

    /// Deterministic sharding (see README): global frame index g is rendered iff g % shard_count == shard_index.
    int shard_index = 0;
    int shard_count = 1;
    int frames_rendered_this_process = 0;
};

/// Fully resolved world configuration as canonical JSON (aliases normalized into structs).
nlohmann::json world_config_to_resolved_json(const WorldConfig& cfg);

/// SHA-256 over UTF-8 bytes, lowercase hex (for dataset_id / optional config digest).
std::string sha256_hex(std::string_view data);

/// Writes `manifest.json` atomically (temp file + rename). Logs on failure; returns false on error.
bool write_dataset_manifest_atomic(const std::filesystem::path& dest_path,
                                   const DatasetManifestParams& params);

} // namespace vf
