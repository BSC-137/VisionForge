#pragma once

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include <nlohmann/json.hpp>

#include "visionforge/config_keys.hpp"
#include "visionforge/vec3.hpp"

struct ScalarRange {
    double min = 0.0;
    double max = 0.0;
};

struct Vec3Range {
    Vec3 min = Vec3(0, 0, 0);
    Vec3 max = Vec3(0, 0, 0);
};

struct AssetEntry {
    std::string name;
    std::string path;
    double weight    = 1.0;
    double scale     = 1.0;
    std::string color = "white";
    std::string label;
    uint32_t class_id = 2;
    double y_offset   = 0.0;
    ScalarRange roughness = {0.5, 0.5};
    ScalarRange metallic  = {0.0, 0.0};

    std::string albedo_map;
    std::string normal_map;
    std::string roughness_map;
    std::string metallic_map;
};

struct CameraKeyframe {
    double t = 0.0;  // normalized frame fraction in [0, 1]
    Vec3   pos;
};

struct WorldConfig {
    struct Render {
        int width = 1280;
        int height = 720;
        int spp = 24;
        int max_depth = 10;
        double exposure = 6.5;
        double sky_gain = 45.0;
        bool preview = false;
        unsigned seed = 1337;
    } render;

    struct Camera {
        Vec3 lookat = Vec3(0.0, 1.2, 0.0);
        Vec3 up = Vec3(0.0, 1.0, 0.0);
        Vec3Range lookfrom = {Vec3(18.0, 8.0, 24.0), Vec3(18.0, 8.0, 24.0)};
        ScalarRange fov_deg = {35.0, 35.0};
        // Optional keyframe trajectory; when non-empty overrides lookfrom for per-frame positions.
        std::vector<CameraKeyframe> trajectory;
    } camera;

    struct Lighting {
        ScalarRange sun_azimuth_deg = {300.0, 300.0};
        ScalarRange sun_elevation_deg = {12.0, 12.0};
    } lighting;

    struct Terrain {
        double xmin = -22.0;
        double xmax = 22.0;
        double zmin = -22.0;
        double zmax = 22.0;
        double amp = 1.8;
        double scale = 0.14;
        int nx = 96;
        int nz = 96;
    } terrain;

    // Legacy single asset (populated from first entry of assets vector)
    struct Asset {
        std::string obj;
        double scale = 1.0;
        std::string color = "white";
        std::string label = "model";
        uint32_t class_id = 2;
        double y_offset = 0.0;
        ScalarRange roughness = {0.5, 0.5};
        ScalarRange metallic = {0.0, 0.0};
    } asset;

    std::vector<AssetEntry> assets;

    std::string hdr_path;
    double hdr_intensity = 1.0;

    std::string ground_tex;
    double ground_scale = 2.0;

    struct Placement {
        ScalarRange x = {-18.0, 18.0};
        ScalarRange z = {-18.0, 18.0};
        ScalarRange yaw_deg = {0.0, 360.0};
    } placement;

    struct Dataset {
        std::string root = "dataset";
        double train_split = 0.8;
    } dataset;

    struct NodeEntry {
        std::string name;
        std::string asset; // references AssetEntry::name
        Vec3 position = Vec3(0, 0, 0);
        Vec3 rotation = Vec3(0, 0, 0);
        Vec3 scale = Vec3(1, 1, 1);
        bool grounding_constraint = false;
        double y_offset = 0.0;
        std::vector<NodeEntry> children;
    };

    struct Scenario {
        std::string name;
        Camera camera;
        std::vector<NodeEntry> root_nodes;
    };

    std::vector<Scenario> scenarios;
};

struct WorldConfigError : public std::runtime_error {
    explicit WorldConfigError(std::string msg) : std::runtime_error(std::move(msg)) {}
};

struct WorldConfigParseOptions {
    /// When true (recommended for `forge` / `scenario` CLI): unknown JSON keys are rejected;
    /// deprecated aliases (`albedo_path`, `obj`, …) fail with actionable messages.
    bool strict = true;
};

/// Compute camera lookfrom position for a normalized frame fraction `alpha` in [0, 1].
/// If `cam.trajectory` is non-empty, linearly interpolates between the sorted keyframes.
/// Falls back to `cam.lookfrom.max` when no trajectory is defined (preserves existing behaviour).
inline Vec3 interpolate_trajectory(const WorldConfig::Camera& cam, double alpha) {
    if (cam.trajectory.empty()) {
        return cam.lookfrom.max;
    }
    alpha = std::clamp(alpha, 0.0, 1.0);
    const auto& kfs = cam.trajectory;
    if (kfs.size() == 1) return kfs[0].pos;
    if (alpha <= kfs.front().t) return kfs.front().pos;
    if (alpha >= kfs.back().t)  return kfs.back().pos;
    for (size_t i = 0; i + 1 < kfs.size(); ++i) {
        if (alpha >= kfs[i].t && alpha <= kfs[i + 1].t) {
            const double span = kfs[i + 1].t - kfs[i].t;
            const double u    = (span > 0.0) ? (alpha - kfs[i].t) / span : 0.0;
            return kfs[i].pos + u * (kfs[i + 1].pos - kfs[i].pos);
        }
    }
    return kfs.back().pos;
}

namespace vf_world_config_detail {

using json = nlohmann::json;
namespace ck = visionforge::cfg;

inline std::string fmt_allowed_keys(std::initializer_list<std::string_view> keys) {
    std::ostringstream oss;
    bool first = true;
    for (std::string_view k : keys) {
        if (!first) oss << ", ";
        first = false;
        oss << '`' << k << '`';
    }
    return oss.str();
}

/// Validates sibling JSON keys for one object. `deprecated_aliases` keys fail when strict=true.
inline void validate_object_keys(
    const json& j,
    std::initializer_list<std::string_view> canonical_keys,
    const std::unordered_map<std::string, std::string>& deprecated_aliases,
    const std::string& json_path,
    bool strict
) {
    if (!j.is_object()) {
        throw WorldConfigError(json_path + ": expected JSON object");
    }
    std::unordered_set<std::string_view> allowed;
    allowed.reserve(static_cast<size_t>(canonical_keys.size()));
    for (std::string_view k : canonical_keys) {
        allowed.insert(k);
    }
    const std::string allowed_fmt = fmt_allowed_keys(canonical_keys);

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        if (allowed.count(std::string_view(key))) continue;

        auto dep_it = deprecated_aliases.find(key);
        if (dep_it != deprecated_aliases.end()) {
            if (strict) {
                std::ostringstream oss;
                oss << "World config: deprecated key at `" << json_path << "." << key << "`. "
                    << dep_it->second << " Allowed sibling keys include " << allowed_fmt << '.';
                throw WorldConfigError(oss.str());
            }
            continue;
        }

        std::ostringstream oss;
        oss << "World config: unknown key at `" << json_path << "." << key << "`. "
               "Expected only canonical sibling keys: "
            << allowed_fmt << '.';
        throw WorldConfigError(oss.str());
    }
}

inline Vec3 parse_vec3_json(const json& j, const char* field_name) {
    if (!j.is_array() || j.size() != 3) {
        throw WorldConfigError(std::string("Expected [x,y,z] array for field: ") + field_name);
    }
    return Vec3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
}

inline ScalarRange parse_scalar_range(const json& j, const std::string& path, bool strict) {
    if (j.is_number()) {
        const double v = j.get<double>();
        return {v, v};
    }
    if (!j.is_object()) {
        throw WorldConfigError(path + ": expected number or object with min/max");
    }
    if (strict) {
        validate_object_keys(j, {ck::k_min, ck::k_max}, {}, path, true);
    }
    ScalarRange out;
    out.min = j.at(std::string(ck::k_min)).get<double>();
    out.max = j.at(std::string(ck::k_max)).get<double>();
    if (out.min > out.max) std::swap(out.min, out.max);
    return out;
}

inline Vec3Range parse_vec3_range(const json& j, const char* field_name, const std::string& path, bool strict) {
    if (j.is_array()) {
        const Vec3 v = parse_vec3_json(j, field_name);
        return {v, v};
    }
    if (!j.is_object() || !j.contains(std::string(ck::k_min)) || !j.contains(std::string(ck::k_max))) {
        throw WorldConfigError(std::string(field_name) + ": expected Vec3 array or range object with min/max");
    }
    if (strict) {
        validate_object_keys(j, {ck::k_min, ck::k_max}, {}, path, true);
    }
    Vec3Range out;
    out.min = parse_vec3_json(j.at(std::string(ck::k_min)), field_name);
    out.max = parse_vec3_json(j.at(std::string(ck::k_max)), field_name);
    if (out.min.x > out.max.x) std::swap(out.min.x, out.max.x);
    if (out.min.y > out.max.y) std::swap(out.min.y, out.max.y);
    if (out.min.z > out.max.z) std::swap(out.min.z, out.max.z);
    return out;
}

inline void warn_deprecated_asset_texture_alias(std::string_view deprecated_key, std::string_view canonical_key) {
    std::cerr << "[world.json warning] `" << deprecated_key << "` is deprecated; use `" << canonical_key << "`.\n";
}

inline void warn_deprecated_mesh_obj() {
    std::cerr << "[world.json warning] `obj` is deprecated as a mesh path field; use `path`.\n";
}

inline AssetEntry parse_asset_entry(const json& a, const std::string& path_prefix, bool strict) {
    static const std::unordered_map<std::string, std::string> k_deprecated_asset_keys = {
        {std::string(ck::k_obj),
         "`obj` is deprecated as a mesh path field; use `path`."},
        {std::string(ck::k_albedo_path),
         "`albedo_path` is deprecated; use `albedo_map`."},
        {std::string(ck::k_normal_path),
         "`normal_path` is deprecated; use `normal_map`."},
        {std::string(ck::k_roughness_path),
         "`roughness_path` is deprecated; use `roughness_map`."},
        {std::string(ck::k_metallic_path),
         "`metallic_path` is deprecated; use `metallic_map`."},
    };

    validate_object_keys(a,
                         {
                             ck::k_name,
                             ck::k_path,
                             ck::k_weight,
                             ck::k_scale,
                             ck::k_color,
                             ck::k_label,
                             ck::k_class_id,
                             ck::k_y_offset,
                             ck::k_roughness,
                             ck::k_metallic,
                             ck::k_albedo_map,
                             ck::k_normal_map,
                             ck::k_roughness_map,
                             ck::k_metallic_map,
                         },
                         k_deprecated_asset_keys,
                         path_prefix,
                         strict);

    AssetEntry e;

    if (a.contains(std::string(ck::k_name))) e.name = a.at(std::string(ck::k_name)).get<std::string>();

    const bool has_path = a.contains(std::string(ck::k_path));
    const bool has_obj  = a.contains(std::string(ck::k_obj));

    if (strict) {
        if (has_path) {
            e.path = a.at(std::string(ck::k_path)).get<std::string>();
        } else if (has_obj) {
            throw WorldConfigError(
                "World config: deprecated key `" + path_prefix + ".obj`. "
                "`obj` is not accepted when strict parsing is enabled; use `path`.");
        }
    } else {
        if (has_path && has_obj) {
            e.path = a.at(std::string(ck::k_path)).get<std::string>();
            const std::string alt = a.at(std::string(ck::k_obj)).get<std::string>();
            if (alt != e.path) {
                std::ostringstream oss;
                oss << "[world.json warning] `" << path_prefix << "`: both `path` and `obj` are set; "
                       "using `path` (`"
                    << e.path << "` != `" << alt << "`).\n";
                std::cerr << oss.str();
            }
            warn_deprecated_mesh_obj();
        } else if (has_path) {
            e.path = a.at(std::string(ck::k_path)).get<std::string>();
        } else if (has_obj) {
            warn_deprecated_mesh_obj();
            e.path = a.at(std::string(ck::k_obj)).get<std::string>();
        }
    }

    if (a.contains(std::string(ck::k_weight))) e.weight = a.at(std::string(ck::k_weight)).get<double>();
    if (a.contains(std::string(ck::k_scale))) e.scale = a.at(std::string(ck::k_scale)).get<double>();
    if (a.contains(std::string(ck::k_color))) e.color = a.at(std::string(ck::k_color)).get<std::string>();
    if (a.contains(std::string(ck::k_label))) e.label = a.at(std::string(ck::k_label)).get<std::string>();
    if (a.contains(std::string(ck::k_class_id))) e.class_id = a.at(std::string(ck::k_class_id)).get<uint32_t>();
    if (a.contains(std::string(ck::k_y_offset))) e.y_offset = a.at(std::string(ck::k_y_offset)).get<double>();
    if (a.contains(std::string(ck::k_roughness)))
        e.roughness = parse_scalar_range(a.at(std::string(ck::k_roughness)), path_prefix + ".roughness", strict);
    if (a.contains(std::string(ck::k_metallic)))
        e.metallic = parse_scalar_range(a.at(std::string(ck::k_metallic)), path_prefix + ".metallic", strict);

    auto read_tex_pair = [&](std::string_view canon, std::string_view legacy, std::string& dest) {
        const std::string cs(canon);
        const std::string ls(legacy);
        if (a.contains(cs)) {
            dest = a.at(cs).get<std::string>();
            return;
        }
        if (!strict && a.contains(ls)) {
            warn_deprecated_asset_texture_alias(legacy, canon);
            dest = a.at(ls).get<std::string>();
        }
    };

    read_tex_pair(ck::k_albedo_map, ck::k_albedo_path, e.albedo_map);
    read_tex_pair(ck::k_normal_map, ck::k_normal_path, e.normal_map);
    read_tex_pair(ck::k_roughness_map, ck::k_roughness_path, e.roughness_map);
    read_tex_pair(ck::k_metallic_map, ck::k_metallic_path, e.metallic_map);

    if (e.name.empty() && !e.path.empty()) {
        auto pos = e.path.find_last_of("/\\");
        e.name = (pos != std::string::npos) ? e.path.substr(pos + 1) : e.path;
    }
    if (e.label.empty()) e.label = e.name.empty() ? "model" : e.name;

    return e;
}

inline void validate_node_tree_asset_refs(const WorldConfig::NodeEntry& node,
                                       const std::string& path_prefix,
                                       const std::unordered_set<std::string>& asset_names) {
    if (!node.asset.empty()) {
        if (!asset_names.count(node.asset)) {
            throw WorldConfigError(
                "World config: `" + path_prefix + ".asset` references unknown asset `" + node.asset +
                "`. Expected an `assets[].name` value defined in this config.");
        }
    }
    for (size_t ci = 0; ci < node.children.size(); ++ci) {
        validate_node_tree_asset_refs(node.children[ci],
                                      path_prefix + ".children[" + std::to_string(ci) + "]",
                                      asset_names);
    }
}

inline void validate_world_config_semantics(WorldConfig& cfg, const WorldConfigParseOptions& opt) {
    (void)opt;
    std::unordered_set<std::string> asset_names;
    for (const auto& a : cfg.assets) asset_names.insert(a.name);

    for (size_t si = 0; si < cfg.scenarios.size(); ++si) {
        const auto& sc = cfg.scenarios[si];
        const std::string sprefix = "scenarios[" + std::to_string(si) + "]";
        for (size_t ni = 0; ni < sc.root_nodes.size(); ++ni) {
            validate_node_tree_asset_refs(sc.root_nodes[ni],
                                           sprefix + ".root_nodes[" + std::to_string(ni) + "]",
                                           asset_names);
        }
    }
}

} // namespace vf_world_config_detail

inline WorldConfig load_world_config(const std::string& path, WorldConfigParseOptions opt = {}) {
    using json = nlohmann::json;
    using namespace vf_world_config_detail;
    namespace ck = visionforge::cfg;

    std::ifstream in(path);
    if (!in) {
        throw WorldConfigError("Failed to open world config: " + path);
    }

    json root;
    try {
        in >> root;
    } catch (const std::exception& ex) {
        throw WorldConfigError(std::string("Failed to parse JSON in world config: ") + path + ": " + ex.what());
    }

    const bool strict = opt.strict;

    validate_object_keys(root,
                         {
                             ck::k_render,
                             ck::k_camera,
                             ck::k_lighting,
                             ck::k_terrain,
                             ck::k_assets,
                             ck::k_asset,
                             ck::k_hdr,
                             ck::k_placement,
                             ck::k_dataset,
                             ck::k_scenarios,
                         },
                         {},
                         "",
                         strict);

    WorldConfig cfg;

    if (root.contains(std::string(ck::k_render))) {
        const auto& r = root.at(std::string(ck::k_render));
        validate_object_keys(r,
                             {
                                 ck::k_width,
                                 ck::k_height,
                                 ck::k_spp,
                                 ck::k_max_depth,
                                 ck::k_exposure,
                                 ck::k_sky_gain,
                                 ck::k_preview,
                                 ck::k_seed,
                             },
                             {},
                             "render",
                             strict);
        if (r.contains(std::string(ck::k_width))) cfg.render.width = r.at(std::string(ck::k_width)).get<int>();
        if (r.contains(std::string(ck::k_height))) cfg.render.height = r.at(std::string(ck::k_height)).get<int>();
        if (r.contains(std::string(ck::k_spp))) cfg.render.spp = r.at(std::string(ck::k_spp)).get<int>();
        if (r.contains(std::string(ck::k_max_depth))) cfg.render.max_depth = r.at(std::string(ck::k_max_depth)).get<int>();
        if (r.contains(std::string(ck::k_exposure))) cfg.render.exposure = r.at(std::string(ck::k_exposure)).get<double>();
        if (r.contains(std::string(ck::k_sky_gain))) cfg.render.sky_gain = r.at(std::string(ck::k_sky_gain)).get<double>();
        if (r.contains(std::string(ck::k_preview))) cfg.render.preview = r.at(std::string(ck::k_preview)).get<bool>();
        if (r.contains(std::string(ck::k_seed))) cfg.render.seed = r.at(std::string(ck::k_seed)).get<unsigned>();
    }

    auto parse_camera_trajectory = [&](const json& c, const std::string& cam_path,
                                        WorldConfig::Camera& cam_out) {
        if (!c.contains(std::string(ck::k_trajectory))) return;
        const auto& arr = c.at(std::string(ck::k_trajectory));
        if (!arr.is_array()) {
            throw WorldConfigError(cam_path + ".trajectory: expected JSON array of keyframes");
        }
        cam_out.trajectory.clear();
        size_t ki = 0;
        for (const auto& kf : arr) {
            const std::string kfpath = cam_path + ".trajectory[" + std::to_string(ki) + "]";
            validate_object_keys(kf, {ck::k_t, ck::k_pos}, {}, kfpath, strict);
            if (!kf.contains(std::string(ck::k_t)))
                throw WorldConfigError(kfpath + ": missing required field `t`");
            if (!kf.contains(std::string(ck::k_pos)))
                throw WorldConfigError(kfpath + ": missing required field `pos`");
            CameraKeyframe ckf;
            ckf.t   = kf.at(std::string(ck::k_t)).get<double>();
            ckf.pos = parse_vec3_json(kf.at(std::string(ck::k_pos)), (kfpath + ".pos").c_str());
            cam_out.trajectory.push_back(ckf);
            ++ki;
        }
        std::stable_sort(cam_out.trajectory.begin(), cam_out.trajectory.end(),
                         [](const CameraKeyframe& a, const CameraKeyframe& b){ return a.t < b.t; });
    };

    if (root.contains(std::string(ck::k_camera))) {
        const auto& c = root.at(std::string(ck::k_camera));
        validate_object_keys(c,
                             {
                                 ck::k_lookat,
                                 ck::k_up,
                                 ck::k_lookfrom,
                                 ck::k_fov_deg,
                                 ck::k_trajectory,
                             },
                             {},
                             "camera",
                             strict);
        if (c.contains(std::string(ck::k_lookat)))
            cfg.camera.lookat = parse_vec3_json(c.at(std::string(ck::k_lookat)), "camera.lookat");
        if (c.contains(std::string(ck::k_up)))
            cfg.camera.up = parse_vec3_json(c.at(std::string(ck::k_up)), "camera.up");
        if (c.contains(std::string(ck::k_lookfrom)))
            cfg.camera.lookfrom =
                parse_vec3_range(c.at(std::string(ck::k_lookfrom)), "camera.lookfrom", "camera.lookfrom", strict);
        if (c.contains(std::string(ck::k_fov_deg)))
            cfg.camera.fov_deg =
                parse_scalar_range(c.at(std::string(ck::k_fov_deg)), "camera.fov_deg", strict);
        parse_camera_trajectory(c, "camera", cfg.camera);
    }

    if (root.contains(std::string(ck::k_lighting))) {
        const auto& l = root.at(std::string(ck::k_lighting));
        validate_object_keys(l,
                             {
                                 ck::k_sun_azimuth_deg,
                                 ck::k_sun_elevation_deg,
                             },
                             {},
                             "lighting",
                             strict);
        if (l.contains(std::string(ck::k_sun_azimuth_deg)))
            cfg.lighting.sun_azimuth_deg =
                parse_scalar_range(l.at(std::string(ck::k_sun_azimuth_deg)), "lighting.sun_azimuth_deg", strict);
        if (l.contains(std::string(ck::k_sun_elevation_deg)))
            cfg.lighting.sun_elevation_deg =
                parse_scalar_range(l.at(std::string(ck::k_sun_elevation_deg)), "lighting.sun_elevation_deg", strict);
    }

    if (root.contains(std::string(ck::k_terrain))) {
        const auto& t = root.at(std::string(ck::k_terrain));
        validate_object_keys(t,
                             {
                                 ck::k_amp,
                                 ck::k_scale,
                                 ck::k_nx,
                                 ck::k_nz,
                                 ck::k_bounds,
                                 ck::k_xmin,
                                 ck::k_xmax,
                                 ck::k_zmin,
                                 ck::k_zmax,
                             },
                             {},
                             "terrain",
                             strict);

        double xmin = cfg.terrain.xmin;
        double xmax = cfg.terrain.xmax;
        double zmin = cfg.terrain.zmin;
        double zmax = cfg.terrain.zmax;

        if (t.contains(std::string(ck::k_bounds))) {
            const auto& b = t.at(std::string(ck::k_bounds));
            validate_object_keys(b,
                                 {
                                     ck::k_xmin,
                                     ck::k_xmax,
                                     ck::k_zmin,
                                     ck::k_zmax,
                                 },
                                 {},
                                 "terrain.bounds",
                                 strict);
            if (b.contains(std::string(ck::k_xmin))) xmin = b.at(std::string(ck::k_xmin)).get<double>();
            if (b.contains(std::string(ck::k_xmax))) xmax = b.at(std::string(ck::k_xmax)).get<double>();
            if (b.contains(std::string(ck::k_zmin))) zmin = b.at(std::string(ck::k_zmin)).get<double>();
            if (b.contains(std::string(ck::k_zmax))) zmax = b.at(std::string(ck::k_zmax)).get<double>();
        }

        const bool flat_any = t.contains(std::string(ck::k_xmin)) || t.contains(std::string(ck::k_xmax))
                           || t.contains(std::string(ck::k_zmin)) || t.contains(std::string(ck::k_zmax));
        if (flat_any && t.contains(std::string(ck::k_bounds)) && !strict) {
            std::cerr << "[world.json warning] `terrain`: both `bounds` and flat xmin/xmax/zmin/zmax are set; "
                         "flat keys override `bounds`.\n";
        }

        if (t.contains(std::string(ck::k_xmin))) xmin = t.at(std::string(ck::k_xmin)).get<double>();
        if (t.contains(std::string(ck::k_xmax))) xmax = t.at(std::string(ck::k_xmax)).get<double>();
        if (t.contains(std::string(ck::k_zmin))) zmin = t.at(std::string(ck::k_zmin)).get<double>();
        if (t.contains(std::string(ck::k_zmax))) zmax = t.at(std::string(ck::k_zmax)).get<double>();

        cfg.terrain.xmin = xmin;
        cfg.terrain.xmax = xmax;
        cfg.terrain.zmin = zmin;
        cfg.terrain.zmax = zmax;

        if (t.contains(std::string(ck::k_amp))) cfg.terrain.amp = t.at(std::string(ck::k_amp)).get<double>();
        if (t.contains(std::string(ck::k_scale))) cfg.terrain.scale = t.at(std::string(ck::k_scale)).get<double>();
        if (t.contains(std::string(ck::k_nx))) cfg.terrain.nx = t.at(std::string(ck::k_nx)).get<int>();
        if (t.contains(std::string(ck::k_nz))) cfg.terrain.nz = t.at(std::string(ck::k_nz)).get<int>();
    }

    if (root.contains(std::string(ck::k_assets)) && root.at(std::string(ck::k_assets)).is_array()) {
        size_t idx = 0;
        for (const auto& elem : root.at(std::string(ck::k_assets))) {
            cfg.assets.push_back(parse_asset_entry(elem, std::string("assets[") + std::to_string(idx) + "]", strict));
            ++idx;
        }
    } else if (root.contains(std::string(ck::k_asset))) {
        cfg.assets.push_back(parse_asset_entry(root.at(std::string(ck::k_asset)), "asset", strict));
    }

    if (!cfg.assets.empty()) {
        const auto& first = cfg.assets[0];
        cfg.asset.obj       = first.path;
        cfg.asset.scale     = first.scale;
        cfg.asset.color     = first.color;
        cfg.asset.label     = first.label;
        cfg.asset.class_id  = first.class_id;
        cfg.asset.y_offset  = first.y_offset;
        cfg.asset.roughness = first.roughness;
        cfg.asset.metallic  = first.metallic;
    }

    if (root.contains(std::string(ck::k_hdr))) {
        const auto& h = root.at(std::string(ck::k_hdr));
        if (h.is_string()) {
            cfg.hdr_path = h.get<std::string>();
        } else if (h.is_object()) {
            validate_object_keys(h,
                                 {
                                     ck::k_path,
                                     ck::k_intensity,
                                 },
                                 {},
                                 "hdr",
                                 strict);
            if (h.contains(std::string(ck::k_path))) cfg.hdr_path = h.at(std::string(ck::k_path)).get<std::string>();
            if (h.contains(std::string(ck::k_intensity)))
                cfg.hdr_intensity = h.at(std::string(ck::k_intensity)).get<double>();
        } else {
            throw WorldConfigError("`hdr` must be a string path or an object with `path` / `intensity`.");
        }
    }

    if (root.contains(std::string(ck::k_placement))) {
        const auto& p = root.at(std::string(ck::k_placement));
        validate_object_keys(p,
                             {
                                 ck::k_x,
                                 ck::k_z,
                                 ck::k_yaw_deg,
                             },
                             {},
                             "placement",
                             strict);
        if (p.contains(std::string(ck::k_x)))
            cfg.placement.x = parse_scalar_range(p.at(std::string(ck::k_x)), "placement.x", strict);
        if (p.contains(std::string(ck::k_z)))
            cfg.placement.z = parse_scalar_range(p.at(std::string(ck::k_z)), "placement.z", strict);
        if (p.contains(std::string(ck::k_yaw_deg)))
            cfg.placement.yaw_deg =
                parse_scalar_range(p.at(std::string(ck::k_yaw_deg)), "placement.yaw_deg", strict);
    }

    if (root.contains(std::string(ck::k_dataset))) {
        const auto& d = root.at(std::string(ck::k_dataset));
        validate_object_keys(d,
                             {
                                 ck::k_root,
                                 ck::k_train_split,
                             },
                             {},
                             "dataset",
                             strict);
        if (d.contains(std::string(ck::k_root))) cfg.dataset.root = d.at(std::string(ck::k_root)).get<std::string>();
        if (d.contains(std::string(ck::k_train_split)))
            cfg.dataset.train_split = d.at(std::string(ck::k_train_split)).get<double>();
    }

    std::function<WorldConfig::NodeEntry(const json&, const std::string&)> parse_node_entry;
    parse_node_entry = [&](const json& j, const std::string& node_path) -> WorldConfig::NodeEntry {
        validate_object_keys(j,
                             {
                                 ck::k_name,
                                 ck::k_asset,
                                 ck::k_position,
                                 ck::k_rotation,
                                 ck::k_scale,
                                 ck::k_grounding_constraint,
                                 ck::k_y_offset,
                                 ck::k_children,
                             },
                             {},
                             node_path,
                             strict);

        WorldConfig::NodeEntry node;
        if (j.contains(std::string(ck::k_name))) node.name = j.at(std::string(ck::k_name)).get<std::string>();
        if (j.contains(std::string(ck::k_asset))) node.asset = j.at(std::string(ck::k_asset)).get<std::string>();
        if (j.contains(std::string(ck::k_position)))
            node.position = parse_vec3_json(j.at(std::string(ck::k_position)), "NodeEntry.position");
        if (j.contains(std::string(ck::k_rotation)))
            node.rotation = parse_vec3_json(j.at(std::string(ck::k_rotation)), "NodeEntry.rotation");
        if (j.contains(std::string(ck::k_scale))) {
            if (j.at(std::string(ck::k_scale)).is_array())
                node.scale = parse_vec3_json(j.at(std::string(ck::k_scale)), "NodeEntry.scale");
            else {
                double s = j.at(std::string(ck::k_scale)).get<double>();
                node.scale = Vec3(s, s, s);
            }
        }
        if (j.contains(std::string(ck::k_grounding_constraint)))
            node.grounding_constraint = j.at(std::string(ck::k_grounding_constraint)).get<bool>();
        if (j.contains(std::string(ck::k_y_offset))) node.y_offset = j.at(std::string(ck::k_y_offset)).get<double>();
        if (j.contains(std::string(ck::k_children)) && j.at(std::string(ck::k_children)).is_array()) {
            size_t ci = 0;
            for (const auto& c : j.at(std::string(ck::k_children))) {
                node.children.push_back(parse_node_entry(c, node_path + ".children[" + std::to_string(ci) + "]"));
                ++ci;
            }
        }
        return node;
    };

    if (root.contains(std::string(ck::k_scenarios)) && root.at(std::string(ck::k_scenarios)).is_array()) {
        const auto& arr = root.at(std::string(ck::k_scenarios));
        size_t si = 0;
        for (const auto& s : arr) {
            const std::string sp = "scenarios[" + std::to_string(si) + "]";
            validate_object_keys(s,
                                 {
                                     ck::k_name,
                                     ck::k_camera,
                                     ck::k_root_nodes,
                                 },
                                 {},
                                 sp,
                                 strict);

            if (!s.contains(std::string(ck::k_name))) {
                throw WorldConfigError("World config: `" + sp + "` missing required field `name`.");
            }

            WorldConfig::Scenario scenario;
            scenario.name = s.at(std::string(ck::k_name)).get<std::string>();
            if (scenario.name.empty()) {
                throw WorldConfigError("World config: `" + sp + ".name` must be a non-empty string.");
            }

            scenario.camera = cfg.camera;
            if (s.contains(std::string(ck::k_camera))) {
                const auto& c = s.at(std::string(ck::k_camera));
                validate_object_keys(c,
                                     {
                                         ck::k_lookat,
                                         ck::k_up,
                                         ck::k_lookfrom,
                                         ck::k_fov_deg,
                                         ck::k_trajectory,
                                     },
                                     {},
                                     sp + ".camera",
                                     strict);
                if (c.contains(std::string(ck::k_lookat)))
                    scenario.camera.lookat = parse_vec3_json(c.at(std::string(ck::k_lookat)), "camera.lookat");
                if (c.contains(std::string(ck::k_up)))
                    scenario.camera.up = parse_vec3_json(c.at(std::string(ck::k_up)), "camera.up");
                if (c.contains(std::string(ck::k_lookfrom)))
                    scenario.camera.lookfrom =
                        parse_vec3_range(c.at(std::string(ck::k_lookfrom)), "camera.lookfrom",
                                         sp + ".camera.lookfrom", strict);
                if (c.contains(std::string(ck::k_fov_deg)))
                    scenario.camera.fov_deg =
                        parse_scalar_range(c.at(std::string(ck::k_fov_deg)), sp + ".camera.fov_deg", strict);
                parse_camera_trajectory(c, sp + ".camera", scenario.camera);
            }

            if (s.contains(std::string(ck::k_root_nodes)) && s.at(std::string(ck::k_root_nodes)).is_array()) {
                size_t ni = 0;
                for (const auto& n : s.at(std::string(ck::k_root_nodes))) {
                    scenario.root_nodes.push_back(
                        parse_node_entry(n, sp + ".root_nodes[" + std::to_string(ni) + "]"));
                    ++ni;
                }
            }

            cfg.scenarios.push_back(scenario);
            ++si;
        }
    }

    cfg.dataset.train_split = std::clamp(cfg.dataset.train_split, 0.0, 1.0);

    if (cfg.assets.empty()) {
        throw WorldConfigError("World config: `assets` (or legacy `asset`) must define at least one mesh entry.");
    }
    if (cfg.assets[0].path.empty()) {
        throw WorldConfigError(
            "World config: asset entries must resolve to a mesh path via `path` (or deprecated `obj` when strict "
            "parsing is disabled).");
    }

    validate_world_config_semantics(cfg, opt);

    return cfg;
}
