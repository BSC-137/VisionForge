#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include "visionforge/vec3.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/hittable_list.hpp"
#include "visionforge/transform.hpp"
#include "visionforge/terrain.hpp"
#include "visionforge/tag.hpp"
#include "visionforge/placement.hpp" // For snap_y and SlopeAlign
#include "visionforge/mesh.hpp"
#include "visionforge/bvh.hpp"

struct Transform {
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0); // Yaw, Pitch, Roll (Y, X, Z) in degrees. Actually standard is Yaw-Pitch-Roll.
    Vec3 scale = Vec3(1, 1, 1);
};

class SceneNode : public std::enable_shared_from_this<SceneNode> {
public:
    std::string name;
    Transform local_transform;
    Transform world_transform;
    
    std::shared_ptr<Hittable> object = nullptr;
    std::vector<std::shared_ptr<SceneNode>> children;
    SceneNode* parent = nullptr;
    
    bool grounding_constraint = false;
    double y_offset = 0.0;
    
    uint32_t instance_id = 0;
    uint32_t class_id = 0;
    std::string label = ""; // Kept owned by SceneNode instance outliving rendering
    
    SceneNode(const std::string& n) : name(n) {}

private:
    // Cache for flat_pack() so static scenarios don't rebuild BVHs every frame.
    // Safe because it depends only on node object + world transform + tagging metadata.
    mutable bool cache_valid_ = false;
    mutable std::shared_ptr<Hittable> cached_packed_ = nullptr;
    mutable std::shared_ptr<Hittable> cached_object_ = nullptr;
    mutable Transform cached_world_ {};
    mutable bool cached_grounding_ = false;
    mutable double cached_y_offset_ = 0.0;
    mutable Vec3 cached_terrain_normal_ = Vec3(0, 1, 0);
    mutable uint32_t cached_instance_id_ = 0;
    mutable uint32_t cached_class_id_ = 0;
    mutable const char* cached_label_ptr_ = nullptr;

    static inline bool nearly_equal(double a, double b, double eps = 1e-12) {
        return std::fabs(a - b) <= eps;
    }
    static inline bool nearly_equal_vec3(const Vec3& a, const Vec3& b, double eps = 1e-12) {
        return nearly_equal(a.x, b.x, eps) && nearly_equal(a.y, b.y, eps) && nearly_equal(a.z, b.z, eps);
    }
    static inline bool same_transform(const Transform& a, const Transform& b, double eps = 1e-12) {
        return nearly_equal_vec3(a.position, b.position, eps)
            && nearly_equal_vec3(a.rotation, b.rotation, eps)
            && nearly_equal_vec3(a.scale, b.scale, eps);
    }

    inline bool cache_matches() const {
        return cache_valid_
            && cached_object_ == object
            && same_transform(cached_world_, world_transform)
            && cached_grounding_ == grounding_constraint
            && nearly_equal(cached_y_offset_, y_offset)
            && nearly_equal_vec3(cached_terrain_normal_, terrain_normal)
            && cached_instance_id_ == instance_id
            && cached_class_id_ == class_id
            && cached_label_ptr_ == label.c_str()
            && cached_packed_ != nullptr;
    }
    inline void update_cache(std::shared_ptr<Hittable> packed) const {
        cache_valid_ = true;
        cached_packed_ = std::move(packed);
        cached_object_ = object;
        cached_world_ = world_transform;
        cached_grounding_ = grounding_constraint;
        cached_y_offset_ = y_offset;
        cached_terrain_normal_ = terrain_normal;
        cached_instance_id_ = instance_id;
        cached_class_id_ = class_id;
        cached_label_ptr_ = label.c_str();
    }

public:
    void add_child(std::shared_ptr<SceneNode> child) {
        if (child) {
            child->parent = this;
            children.push_back(child);
        }
    }

    Vec3 terrain_normal = Vec3(0, 1, 0);

    void update_transforms(const Transform& parent_world = Transform{}) {
        double yaw_rad = parent_world.rotation.y * PI / 180.0;
        double c = std::cos(yaw_rad);
        double s = std::sin(yaw_rad);
        
        Vec3 scaled_pos = Vec3(
            local_transform.position.x * parent_world.scale.x,
            local_transform.position.y * parent_world.scale.y,
            local_transform.position.z * parent_world.scale.z
        );
        
        Vec3 rotated_pos = Vec3(
            scaled_pos.x * c + scaled_pos.z * s,
            scaled_pos.y,
            -scaled_pos.x * s + scaled_pos.z * c
        );

        world_transform.position = parent_world.position + rotated_pos;
        world_transform.rotation = parent_world.rotation + local_transform.rotation;
        world_transform.scale = Vec3(
            parent_world.scale.x * local_transform.scale.x,
            parent_world.scale.y * local_transform.scale.y,
            parent_world.scale.z * local_transform.scale.z
        );

        for (auto& child : children) {
            child->update_transforms(world_transform);
        }
    }

    void apply_grounding(const HeightField& hf, double parent_delta_y = 0.0) {
        world_transform.position.y += parent_delta_y;
        double my_delta_y = parent_delta_y;

        if (grounding_constraint) {
            auto terrain_sample = hf.get_terrain_height_and_normal(world_transform.position.x, world_transform.position.z);
            double new_y = terrain_sample.height;
            my_delta_y += (new_y - world_transform.position.y);
            world_transform.position.y = new_y;
            terrain_normal = terrain_sample.normal;
        } else {
            terrain_normal = Vec3(0, 1, 0);
        }

        for (auto& child : children) {
            child->apply_grounding(hf, my_delta_y);
        }
    }

    void flat_pack(HittableList& out_list) {
        if (object) {
            if (cache_matches()) {
                out_list.add(cached_packed_);
            } else {
            std::shared_ptr<Mesh> base_mesh = std::dynamic_pointer_cast<Mesh>(object);

            if (base_mesh) {
                // Bake transforms directly into the primitive vertices 
                auto baked_data = std::make_shared<MeshData>();
                baked_data->positions.reserve(base_mesh->data->positions.size());
                baked_data->normals.reserve(base_mesh->data->normals.size());
                baked_data->texcoords = base_mesh->data->texcoords;

                // Precompute world transforms
                double rx = world_transform.rotation.x * PI / 180.0;
                double ry = world_transform.rotation.y * PI / 180.0;
                double rz = world_transform.rotation.z * PI / 180.0;

                double cx = std::cos(rx), sx = std::sin(rx);
                double cy = std::cos(ry), sy = std::sin(ry);
                double cz = std::cos(rz), sz = std::sin(rz);

                // Rotation matrices
                auto rot_x = [cx, sx](Vec3& p) {
                    double y = p.y * cx - p.z * sx;
                    double z = p.y * sx + p.z * cx;
                    p.y = y; p.z = z;
                };

                auto rot_y = [cy, sy](Vec3& p) {
                    double x = p.x * cy + p.z * sy;
                    double z = -p.x * sy + p.z * cy;
                    p.x = x; p.z = z;
                };

                auto rot_z = [cz, sz](Vec3& p) {
                    double x = p.x * cz - p.y * sz;
                    double y = p.x * sz + p.y * cz;
                    p.x = x; p.y = y;
                };

                Mat3 t_rot;
                if (grounding_constraint) {
                    SlopeRotation sr = slope_rotation_from_normal(terrain_normal);
                    t_rot.cols[0] = Vec3(sr.m[0][0], sr.m[1][0], sr.m[2][0]);
                    t_rot.cols[1] = Vec3(sr.m[0][1], sr.m[1][1], sr.m[2][1]);
                    t_rot.cols[2] = Vec3(sr.m[0][2], sr.m[1][2], sr.m[2][2]);
                }

                auto apply_transforms = [&](const Vec3& p_in, Vec3& p_out, bool is_normal) {
                    Vec3 p = p_in;
                    if (!is_normal) {
                        p.x *= world_transform.scale.x;
                        p.y *= world_transform.scale.y;
                        p.z *= world_transform.scale.z;
                    }
                    else if (world_transform.scale.x != 1 || world_transform.scale.y != 1 || world_transform.scale.z != 1) {
                         p.x *= 1.0 / world_transform.scale.x;
                         p.y *= 1.0 / world_transform.scale.y;
                         p.z *= 1.0 / world_transform.scale.z;
                         p = normalize(p);
                    }

                    if (world_transform.rotation.x != 0) rot_x(p);
                    if (world_transform.rotation.y != 0) rot_y(p);
                    if (world_transform.rotation.z != 0) rot_z(p);

                    if (grounding_constraint) {
                         p = t_rot * p; 
                    }

                    if (!is_normal) p += world_transform.position;
                    p_out = p;
                };
                
                Vec3 inv_scale(1.0/world_transform.scale.x, 1.0/world_transform.scale.y, 1.0/world_transform.scale.z);

                for (const auto& p : base_mesh->data->positions) {
                    Vec3 out;
                    apply_transforms(p, out, false);
                    baked_data->positions.push_back(out);
                }

                for (const auto& n : base_mesh->data->normals) {
                    Vec3 out;
                    apply_transforms(n, out, true);
                    baked_data->normals.push_back(out);
                }

                double snap_offset = 0;
                if (grounding_constraint) {
                    Vec3 minv(1e9, 1e9, 1e9), maxv(-1e9, -1e9, -1e9);
                    for (const auto& p : baked_data->positions) {
                        if (p.x == 0 && p.y == 0 && p.z == 0) continue; // skip 0th dummy vec
                        minv = min_vec(minv, p);
                        maxv = max_vec(maxv, p);
                    }
                    snap_offset = snap_y(world_transform.position.y, maxv.y - minv.y, 0.0) + y_offset - world_transform.position.y;
                    for (auto& p : baked_data->positions) {
                        if (p.x == 0 && p.y == 0 && p.z == 0) continue;
                        p.y += snap_offset;
                    }
                } else {
                    snap_offset = y_offset;
                    for (auto& p : baked_data->positions) {
                        if (p.x == 0 && p.y == 0 && p.z == 0) continue;
                        p.y += snap_offset;
                    }
                }

                auto shared_data = std::const_pointer_cast<const MeshData>(baked_data);
                std::vector<std::shared_ptr<Hittable>> new_tris;

                for (const auto& tri_ptr : base_mesh->original_tris) {
                    auto old_tri = std::dynamic_pointer_cast<MeshTriangle>(tri_ptr);
                    auto new_tri = std::make_shared<MeshTriangle>(
                        shared_data,
                        old_tri->i0, old_tri->i1, old_tri->i2,
                        old_tri->n0, old_tri->n1, old_tri->n2,
                        old_tri->t0, old_tri->t1, old_tri->t2,
                        old_tri->has_normals, old_tri->has_texcoords, old_tri->mat
                    );

                    if (instance_id > 0) {
                        ObjectInfo info{instance_id, class_id, label.c_str()};
                        auto tag = std::make_shared<Tag>(new_tri, info);
                        new_tris.push_back(tag);
                    } else {
                        new_tris.push_back(new_tri);
                    }
                }

                auto new_bvh = std::make_shared<BVHNode>(new_tris, 0, new_tris.size());
                out_list.add(new_bvh);
                update_cache(new_bvh);
            } else {
                // Fallback for non-mesh types
                std::shared_ptr<Hittable> wrapper = object;

                if (world_transform.scale.x != 1.0 || world_transform.scale.y != 1.0 || world_transform.scale.z != 1.0) {
                    wrapper = std::make_shared<Scale>(wrapper, world_transform.scale);
                }

                if (world_transform.rotation.z != 0.0) wrapper = std::make_shared<RotateZ>(wrapper, world_transform.rotation.z);
                if (world_transform.rotation.x != 0.0) wrapper = std::make_shared<RotateX>(wrapper, world_transform.rotation.x);
                if (world_transform.rotation.y != 0.0) wrapper = std::make_shared<RotateY>(wrapper, world_transform.rotation.y);

                Vec3 final_pos = world_transform.position;
                
                if (grounding_constraint) {
                    wrapper = std::make_shared<SlopeAlign>(wrapper, terrain_normal);
                    AABB aabb;
                    if (wrapper->bounding_box(aabb)) {
                        final_pos.y = snap_y(final_pos.y, aabb.max().y - aabb.min().y, 0.0) + y_offset;
                    }
                } else {
                    final_pos.y += y_offset;
                }

                wrapper = std::make_shared<Translate>(wrapper, final_pos);

                if (instance_id > 0) {
                    ObjectInfo info{instance_id, class_id, label.c_str()};
                    wrapper = std::make_shared<Tag>(wrapper, info);
                }

                out_list.add(wrapper);
                update_cache(wrapper);
            }
            }
        }

        for (auto& child : children) {
            child->flat_pack(out_list);
        }
    }
};
