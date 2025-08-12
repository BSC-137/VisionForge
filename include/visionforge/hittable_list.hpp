#pragma once
#include <memory>
#include <vector>
#include "hittable.hpp"

class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    void clear() { objects.clear(); }
    void add(std::shared_ptr<Hittable> obj) { objects.push_back(std::move(obj)); }

    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;

        for (const auto& obj : objects) {
            if (obj->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    bool bounding_box(AABB& out_box) const override {
        if (objects.empty()) return false;
        AABB temp;
        bool first = true;
        for (const auto& o : objects) {
            if (!o->bounding_box(temp)) return false;
            out_box = first ? temp : surrounding_box(out_box, temp);
            first = false;
        }
        return true;
    }
};
