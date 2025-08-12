#pragma once
#include <algorithm>
#include <memory>
#include <vector>
#include "hittable.hpp"
#include "aabb.hpp"
#include "vec3.hpp"

class BVHNode : public Hittable {
public:
    std::shared_ptr<Hittable> left;
    std::shared_ptr<Hittable> right;
    AABB box;

    BVHNode() {}

    BVHNode(std::vector<std::shared_ptr<Hittable>>& src, size_t start, size_t end) {
        size_t span = end - start;
        int axis = int(3.0 * random_double()); // 0,1,2

        auto box_less = [axis](const std::shared_ptr<Hittable>& a,
                               const std::shared_ptr<Hittable>& b) {
            AABB ba, bb;
            if (!a->bounding_box(ba) || !b->bounding_box(bb))
                std::cerr << "BVH: missing bounding_box()\n";
            if (axis == 0) return ba.min().x < bb.min().x;
            if (axis == 1) return ba.min().y < bb.min().y;
            return ba.min().z < bb.min().z;
        };

        if (span == 1) {
            left = right = src[start];
        } else if (span == 2) {
            if (box_less(src[start], src[start+1])) {
                left = src[start]; right = src[start+1];
            } else {
                left = src[start+1]; right = src[start];
            }
        } else {
            std::sort(src.begin() + start, src.begin() + end, box_less);
            size_t mid = start + span/2;
            left  = std::make_shared<BVHNode>(src, start, mid);
            right = std::make_shared<BVHNode>(src, mid, end);
        }

        AABB bl, br;
        left->bounding_box(bl);
        right->bounding_box(br);
        box = surrounding_box(bl, br);
    }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        if (!box.hit(r, t_min, t_max)) return false;

        bool hit_left  = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
        return hit_left || hit_right;
    }

    bool bounding_box(AABB& out_box) const override {
        out_box = box;
        return true;
    }
};
