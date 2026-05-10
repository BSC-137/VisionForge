#pragma once

#include <string_view>

namespace visionforge::cfg {

/// Canonical JSON keys for VisionForge world / forge configs.
/// Asset texture paths must use *_map suffix only (`albedo_map`, `normal_map`, `roughness_map`,
/// `metallic_map`). Legacy *_path aliases are accepted only when strict parsing is disabled.

// Root document
inline constexpr std::string_view k_render    = "render";
inline constexpr std::string_view k_camera    = "camera";
inline constexpr std::string_view k_lighting   = "lighting";
inline constexpr std::string_view k_terrain    = "terrain";
inline constexpr std::string_view k_assets     = "assets";
inline constexpr std::string_view k_asset      = "asset"; // legacy single-object form
inline constexpr std::string_view k_hdr        = "hdr";
inline constexpr std::string_view k_placement  = "placement";
inline constexpr std::string_view k_dataset    = "dataset";
inline constexpr std::string_view k_scenarios  = "scenarios";

// render
inline constexpr std::string_view k_width      = "width";
inline constexpr std::string_view k_height     = "height";
inline constexpr std::string_view k_spp        = "spp";
inline constexpr std::string_view k_max_depth  = "max_depth";
inline constexpr std::string_view k_exposure   = "exposure";
inline constexpr std::string_view k_sky_gain   = "sky_gain";
inline constexpr std::string_view k_preview    = "preview";
inline constexpr std::string_view k_seed       = "seed";

// camera
inline constexpr std::string_view k_lookat     = "lookat";
inline constexpr std::string_view k_up           = "up";
inline constexpr std::string_view k_lookfrom   = "lookfrom";
inline constexpr std::string_view k_fov_deg    = "fov_deg";

// lighting
inline constexpr std::string_view k_sun_azimuth_deg   = "sun_azimuth_deg";
inline constexpr std::string_view k_sun_elevation_deg = "sun_elevation_deg";

// terrain — canonical bounds may be flat on `terrain` OR nested under `bounds` (both supported).
inline constexpr std::string_view k_amp    = "amp";
inline constexpr std::string_view k_scale  = "scale";
inline constexpr std::string_view k_nx     = "nx";
inline constexpr std::string_view k_nz     = "nz";
inline constexpr std::string_view k_bounds = "bounds";
inline constexpr std::string_view k_xmin   = "xmin";
inline constexpr std::string_view k_xmax   = "xmax";
inline constexpr std::string_view k_zmin   = "zmin";
inline constexpr std::string_view k_zmax   = "zmax";

// asset entry
inline constexpr std::string_view k_name          = "name";
inline constexpr std::string_view k_path            = "path";
inline constexpr std::string_view k_obj             = "obj"; // legacy mesh path alias
inline constexpr std::string_view k_weight          = "weight";
inline constexpr std::string_view k_color           = "color";
inline constexpr std::string_view k_label           = "label";
inline constexpr std::string_view k_class_id        = "class_id";
inline constexpr std::string_view k_y_offset        = "y_offset";
inline constexpr std::string_view k_roughness       = "roughness";
inline constexpr std::string_view k_metallic        = "metallic";
inline constexpr std::string_view k_albedo_map      = "albedo_map";
inline constexpr std::string_view k_normal_map      = "normal_map";
inline constexpr std::string_view k_roughness_map = "roughness_map";
inline constexpr std::string_view k_metallic_map  = "metallic_map";

// Deprecated texture path keys (non-strict only)
inline constexpr std::string_view k_albedo_path      = "albedo_path";
inline constexpr std::string_view k_normal_path       = "normal_path";
inline constexpr std::string_view k_roughness_path   = "roughness_path";
inline constexpr std::string_view k_metallic_path    = "metallic_path";

// hdr object form
inline constexpr std::string_view k_intensity = "intensity";

// placement / dataset
inline constexpr std::string_view k_root         = "root";
inline constexpr std::string_view k_train_split  = "train_split";
inline constexpr std::string_view k_yaw_deg      = "yaw_deg";
inline constexpr std::string_view k_x            = "x";
inline constexpr std::string_view k_z            = "z";

// scenario
inline constexpr std::string_view k_root_nodes = "root_nodes";

// node tree
inline constexpr std::string_view k_position   = "position";
inline constexpr std::string_view k_rotation   = "rotation";
inline constexpr std::string_view k_children   = "children";
inline constexpr std::string_view k_grounding_constraint = "grounding_constraint";

// scalar / vec3 range helpers
inline constexpr std::string_view k_min = "min";
inline constexpr std::string_view k_max = "max";

} // namespace visionforge::cfg
