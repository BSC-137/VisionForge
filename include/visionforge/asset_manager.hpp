#pragma once
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <unordered_map>
#include <iostream>
#include <functional>

#include "visionforge/vec3.hpp"
#include "visionforge/world_config.hpp"
#include "visionforge/mesh.hpp"
#include "visionforge/pbr_material.hpp"
#include "visionforge/transform.hpp"
#include "visionforge/image_texture.hpp"
#include "visionforge/placement.hpp"
#include "visionforge/tag.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/aabb.hpp"

struct LoadedAsset {
    std::string name;
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<PBRMaterial> material;
    AssetEntry config;

    std::shared_ptr<RotateY>    rotate_node;
    std::shared_ptr<SlopeAlign> slope_node;
    std::shared_ptr<Translate>  translate_node;
    std::shared_ptr<Tag>        tag_node;
};

using ColorParseFn = std::function<Vec3(const std::string&, bool&, std::string&)>;

class AssetManager {
public:
    bool load_all(const std::vector<AssetEntry>& entries, const ColorParseFn& parse_color) {
        assets_.clear();

        std::unordered_map<std::string, std::shared_ptr<Mesh>> mesh_cache;
        std::unordered_map<std::string, std::shared_ptr<ImageTexture>> tex_cache;

        auto load_tex = [&](const std::string& p) -> std::shared_ptr<ImageTexture> {
            if (p.empty()) return nullptr;
            auto it = tex_cache.find(p);
            if (it != tex_cache.end()) return it->second;
            auto tex = std::make_shared<ImageTexture>();
            if (tex->load(p)) {
                tex_cache[p] = tex;
                return tex;
            }
            return nullptr;
        };

        std::vector<double> weights;
        for (const auto& entry : entries) {
            LoadedAsset la;
            la.name   = entry.name;
            la.config = entry;

            bool color_ok = true;
            std::string unused;
            Vec3 col = parse_color(entry.color, color_ok, unused);
            la.material = std::make_shared<PBRMaterial>(col, entry.roughness.min, entry.metallic.min);

            la.material->albedo_map    = load_tex(entry.albedo_map);
            la.material->normal_map    = load_tex(entry.normal_map);
            la.material->roughness_map = load_tex(entry.roughness_map);
            la.material->metallic_map  = load_tex(entry.metallic_map);

            auto it = mesh_cache.find(entry.path);
            if (it != mesh_cache.end()) {
                la.mesh = it->second;
            } else {
                la.mesh = Mesh::from_obj(entry.path, la.material, Vec3(0, 0, 0), entry.scale);
                if (!la.mesh) {
                    std::cerr << "AssetManager: failed to load mesh " << entry.path << "\n";
                    return false;
                }
                mesh_cache[entry.path] = la.mesh;
            }

            la.rotate_node    = std::make_shared<RotateY>(la.mesh, 0.0);
            la.slope_node     = std::make_shared<SlopeAlign>(la.rotate_node, Vec3(0, 1, 0));
            la.translate_node = std::make_shared<Translate>(la.slope_node, Vec3(0, 0, 0));
            ObjectInfo info{1u, entry.class_id, entry.label.c_str()};
            la.tag_node       = std::make_shared<Tag>(la.translate_node, info);

            weights.push_back(std::max(entry.weight, 0.001));
            assets_.push_back(std::move(la));
        }

        if (!assets_.empty())
            weight_dist_ = std::discrete_distribution<int>(weights.begin(), weights.end());

        std::cerr << "AssetManager: loaded " << assets_.size() << " assets ("
                  << mesh_cache.size() << " unique meshes)\n";
        return true;
    }

    LoadedAsset* find_by_name(const std::string& name) {
        for (auto& asset : assets_) {
            if (asset.name == name || asset.config.name == name) {
                return &asset;
            }
        }
        return nullptr;
    }

    LoadedAsset& select(std::mt19937& rng) {
        return assets_[weight_dist_(rng)];
    }

    const LoadedAsset& operator[](size_t i) const { return assets_[i]; }
    LoadedAsset& operator[](size_t i) { return assets_[i]; }
    size_t size() const { return assets_.size(); }
    bool empty() const { return assets_.empty(); }

private:
    std::vector<LoadedAsset> assets_;
    std::discrete_distribution<int> weight_dist_;
};
