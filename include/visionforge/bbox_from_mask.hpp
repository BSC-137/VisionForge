#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include "visionforge/passes.hpp"

struct DetectedBox {
    uint32_t instance_id, class_id;
    std::string label;
    int x0,y0,x1,y1; // inclusive
};

struct IdRegistry {
    std::unordered_map<uint32_t, std::string> id_to_label;
    std::unordered_map<uint32_t, uint32_t>    id_to_class;
};

inline std::vector<DetectedBox>
boxes_from_mask(const GBuffer& g, const IdRegistry& reg) {
    struct B { int x0,y0,x1,y1; uint32_t cls; std::string lab; };
    std::unordered_map<uint32_t, B> acc;
    for (int y=0; y<g.height; ++y) for (int x=0; x<g.width; ++x) {
        uint32_t id = g.inst_id[y*g.width + x];
        if (id==0) continue;
        auto it = acc.find(id);
        if (it==acc.end()){
            B b; b.x0=b.x1=x; b.y0=b.y1=y;
            b.cls = reg.id_to_class.count(id)? reg.id_to_class.at(id) : 0u;
            b.lab = reg.id_to_label.count(id)? reg.id_to_label.at(id) : "unknown";
            acc.emplace(id,b);
        } else {
            it->second.x0 = std::min(it->second.x0, x);
            it->second.y0 = std::min(it->second.y0, y);
            it->second.x1 = std::max(it->second.x1, x);
            it->second.y1 = std::max(it->second.y1, y);
        }
    }
    std::vector<DetectedBox> out; out.reserve(acc.size());
    for (auto& kv : acc){
        out.push_back({ kv.first, kv.second.cls, kv.second.lab,
                        kv.second.x0, kv.second.y0, kv.second.x1, kv.second.y1 });
    }
    return out;
}

inline void write_boxes_csv(const std::string& path,
                            const std::vector<DetectedBox>& boxes, int W, int H)
{
    std::ofstream c(path);
    c << "instance_id,class_id,label,xmin,ymin,xmax,ymax,width,height\n";
    for (auto& b : boxes)
        c << b.instance_id << "," << b.class_id << "," << b.label << ","
          << b.x0 << "," << b.y0 << "," << b.x1 << "," << b.y1 << ","
          << W << "," << H << "\n";
}

inline void write_boxes_json(const std::string& path,
                             const std::vector<DetectedBox>& boxes, int W, int H)
{
    std::ofstream j(path);
    j << "{\n  \"image_width\": " << W << ",\n  \"image_height\": " << H << ",\n  \"boxes\": [\n";
    for (size_t i=0;i<boxes.size();++i){
        const auto& b=boxes[i];
        j << "    {\"instance_id\": "<<b.instance_id
          <<", \"class_id\": "<<b.class_id
          <<", \"label\": \""<<b.label<<"\""
          <<", \"xmin\": "<<b.x0<<", \"ymin\": "<<b.y0
          <<", \"xmax\": "<<b.x1<<", \"ymax\": "<<b.y1<<"}";
        if (i+1<boxes.size()) j << ",";
        j << "\n";
    }
    j << "  ]\n}\n";
}
