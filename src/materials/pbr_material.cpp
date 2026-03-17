#include "visionforge/pbr_material.hpp"
#include "visionforge/image_texture.hpp"
#include "visionforge/mesh_triangle.hpp"

bool PBRMaterial::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    bool has_textures = (albedo_map != nullptr) || (roughness_map != nullptr) || (metallic_map != nullptr) || (normal_map != nullptr);

    if (!has_textures) {
        // Fast path for untextured materials
        Vec3 out_dir;
        const Vec3 V = normalize(-r_in.direction);
        Vec3 t, b;
        vf_pbr::build_onb(rec.normal, t, b);
        
        const Vec3 dielectric_F0(0.04, 0.04, 0.04);
        const Vec3 F0 = dielectric_F0 * (1.0 - metallic) + base_color * metallic;
        const double spec_weight = 0.25 + 0.75 * metallic;

        if (random_double() < spec_weight) {
            const double a = std::max(0.02, roughness * roughness);
            Vec3 H = vf_pbr::sample_ggx_half_vector(t, b, rec.normal, a * a);
            const double VdotH = dot(V, H);
            if (VdotH < 0.0) H = -H;
            out_dir = reflect(-V, H);
            const double NdotO = dot(out_dir, rec.normal);
            if (NdotO <= 0.0) out_dir = vf_pbr::sample_cosine_hemisphere(rec.normal, t, b);
            attenuation = vf_pbr::fresnel_schlick(std::fabs(dot(V, H)), F0);
        } else {
            out_dir = vf_pbr::sample_cosine_hemisphere(rec.normal, t, b);
            attenuation = (1.0 - metallic) * base_color;
        }
        scattered = Ray(rec.point, out_dir, r_in.time);
        return true;
    }

    Vec3 color_val = base_color;
    double rough_val = roughness;
    double metal_val = metallic;
    Vec3 N = rec.normal;

    double tex_u = 0.0;
    double tex_v = 0.0;
    if (const auto* tri = dynamic_cast<const MeshTriangle*>(rec.hit_object)) {
        if (tri->has_texcoords && tri->t0 < tri->data->texcoords.size()) {
            const double bu = static_cast<double>(rec.bary_u);
            const double bv = static_cast<double>(rec.bary_v);
            const double w = 1.0 - bu - bv;
            tex_u = w * tri->data->texcoords[tri->t0].x + bu * tri->data->texcoords[tri->t1].x + bv * tri->data->texcoords[tri->t2].x;
            tex_v = w * tri->data->texcoords[tri->t0].y + bu * tri->data->texcoords[tri->t1].y + bv * tri->data->texcoords[tri->t2].y;
        }
    }

    if (albedo_map && albedo_map->valid()) color_val = albedo_map->sample(tex_u, tex_v);
    if (roughness_map && roughness_map->valid()) rough_val = roughness_map->sample(tex_u, tex_v).x;
    if (metallic_map && metallic_map->valid()) metal_val = metallic_map->sample(tex_u, tex_v).x;
    
    if (normal_map && normal_map->valid()) {
        Vec3 sample = normal_map->sample(tex_u, tex_v);
        sample = sample * 2.0 - Vec3(1.0, 1.0, 1.0);
        N = normalize(get_tbn_matrix(rec) * sample);
    }
    
    rough_val = std::clamp(rough_val, 0.0, 1.0);
    metal_val = std::clamp(metal_val, 0.0, 1.0);

    const Vec3 V = normalize(-r_in.direction);

    Vec3 t, b;
    vf_pbr::build_onb(N, t, b);

    const Vec3 dielectric_F0(0.04, 0.04, 0.04);
    const Vec3 F0 = dielectric_F0 * (1.0 - metal_val) + color_val * metal_val;
    const double spec_weight = 0.25 + 0.75 * metal_val;

    Vec3 out_dir;
    if (random_double() < spec_weight) {
        const double a = std::max(0.02, rough_val * rough_val);
        Vec3 H = vf_pbr::sample_ggx_half_vector(t, b, N, a * a);
        const double VdotH = dot(V, H);
        if (VdotH < 0.0) H = -H;
        out_dir = reflect(-V, H);
        const double NdotO = dot(out_dir, N);
        if (NdotO <= 0.0) out_dir = vf_pbr::sample_cosine_hemisphere(N, t, b);
        attenuation = vf_pbr::fresnel_schlick(std::fabs(dot(V, H)), F0);
    } else {
        out_dir = vf_pbr::sample_cosine_hemisphere(N, t, b);
        attenuation = (1.0 - metal_val) * color_val;
    }

    scattered = Ray(rec.point, out_dir, r_in.time);
    return true;
}
