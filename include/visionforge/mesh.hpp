#pragma once
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <fast_obj.h>
#include "visionforge/vec3.hpp"
#include "visionforge/hittable.hpp"
#include "visionforge/material.hpp"
#include "visionforge/aabb.hpp"
#include "visionforge/bvh.hpp"
#include "visionforge/mesh_triangle.hpp"

class Mesh : public Hittable {
public:
    std::shared_ptr<MeshData> data;
    std::shared_ptr<Material> material;
    std::vector<std::shared_ptr<Hittable>> original_tris;
    std::shared_ptr<BVHNode>  bvh_root;
    AABB                      bounds;
    size_t                    tri_count = 0;

    Mesh() : data(std::make_shared<MeshData>()) {}

    static std::shared_ptr<Mesh> from_obj(const std::string& path,
                                          std::shared_ptr<Material> mat,
                                          Vec3 translate = Vec3(0,0,0),
                                          double scale = 1.0)
    {
        auto mesh = std::make_shared<Mesh>();
        mesh->material = std::move(mat);
        if (!mesh->load(path, translate, scale)) return nullptr;
        return mesh;
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (!bvh_root) return false;
        return bvh_root->hit(r, t_min, t_max, rec);
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = bounds;
        return bvh_root != nullptr;
    }

private:
    bool load(const std::string& path, const Vec3& offset, double scale) {
        fastObjMesh* fobj = fast_obj_read(path.c_str());
        if (!fobj) {
            std::cerr << "Mesh: failed to open " << path << "\n";
            return false;
        }

        data->positions.resize(fobj->position_count);
        data->positions[0] = Vec3(0,0,0);
        for (unsigned i = 1; i < fobj->position_count; ++i) {
            float* p = fobj->positions + 3 * i;
            data->positions[i] = Vec3(p[0] * scale, p[1] * scale, p[2] * scale) + offset;
        }

        bool has_normals = fobj->normal_count > 1;
        if (has_normals) {
            data->normals.resize(fobj->normal_count);
            data->normals[0] = Vec3(0,1,0);
            for (unsigned i = 1; i < fobj->normal_count; ++i) {
                float* n = fobj->normals + 3 * i;
                data->normals[i] = normalize(Vec3(n[0], n[1], n[2]));
            }
        }

        bool has_texcoords = fobj->texcoord_count > 1;
        if (has_texcoords) {
            data->texcoords.resize(fobj->texcoord_count);
            data->texcoords[0] = Vec3(0,0,0);
            for (unsigned i = 1; i < fobj->texcoord_count; ++i) {
                float* t = fobj->texcoords + 2 * i;
                data->texcoords[i] = Vec3(t[0], t[1], 0);
            }
        }

        auto shared_data = std::const_pointer_cast<const MeshData>(data);

        std::vector<std::shared_ptr<Hittable>> tris;
        unsigned idx_off = 0;
        for (unsigned f = 0; f < fobj->face_count; ++f) {
            unsigned fv = fobj->face_vertices[f];
            for (unsigned v = 1; v + 1 < fv; ++v) {
                const fastObjIndex& fi0 = fobj->indices[idx_off];
                const fastObjIndex& fi1 = fobj->indices[idx_off + v];
                const fastObjIndex& fi2 = fobj->indices[idx_off + v + 1];

                tris.push_back(std::make_shared<MeshTriangle>(
                    shared_data,
                    fi0.p, fi1.p, fi2.p,
                    fi0.n, fi1.n, fi2.n,
                    fi0.t, fi1.t, fi2.t,
                    has_normals, has_texcoords, material));
            }
            idx_off += fv;
        }

        fast_obj_destroy(fobj);

        tri_count = tris.size();
        original_tris = tris;
        if (tris.empty()) {
            std::cerr << "Mesh: no triangles in " << path << "\n";
            return false;
        }

        bvh_root = std::make_shared<BVHNode>(tris, 0, tris.size());
        bvh_root->bounding_box(bounds);

        std::cerr << "Mesh: loaded " << path
                  << " (" << (data->positions.size() - 1) << " verts, "
                  << tri_count << " tris)\n";
        return true;
    }
};
