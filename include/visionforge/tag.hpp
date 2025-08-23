#pragma once
#include <memory>
#include "hittable.hpp"

class Tag : public Hittable {
public:
    std::shared_ptr<Hittable> child;

    Tag(std::shared_ptr<Hittable> c, const ObjectInfo& info)
        : child(std::move(c)) { this->obj = info; }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        if (!child->hit(r, tmin, tmax, rec)) return false;
        rec.hit_object = this;   // IDs/label come from the tag
        return true;
    }
    bool bounding_box(AABB& out_box) const override {
        return child->bounding_box(out_box);
    }
};
