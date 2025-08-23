#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

struct CocoImage { int id; std::string file_name; int width; int height; };
struct CocoAnn   { int id; int image_id; int category_id;
                   double x, y, w, h; int iscrowd; int area; int instance_id; std::string label; };
struct CocoCat   { int id; std::string name; };

struct CocoWriter {
    std::vector<CocoImage> images;
    std::vector<CocoAnn>   anns;
    std::vector<CocoCat>   cats;
    int next_img_id = 1, next_ann_id = 1;

    void ensure_category(int id, const std::string& name) {
        for (auto& c: cats) if (c.id==id) return;
        cats.push_back({id, name});
    }
    int add_image(const std::string& fname, int W, int H) {
        images.push_back({next_img_id, fname, W, H});
        return next_img_id++;
    }
    void add_box(int image_id, int cat_id, int x0,int y0,int x1,int y1,
                 int instance_id, const std::string& label)
    {
        int w = x1 - x0 + 1, h = y1 - y0 + 1;
        if (w<=0 || h<=0) return;
        anns.push_back({next_ann_id++, image_id, cat_id,
                        double(x0), double(y0), double(w), double(h),
                        0, w*h, instance_id, label});
    }
    void write(const std::string& path) {
        std::ofstream j(path);
        j << "{\n  \"images\": [\n";
        for (size_t i=0;i<images.size();++i){
            auto& im = images[i];
            j << "    {\"id\":"<<im.id<<",\"file_name\":\""<<im.file_name
              <<"\",\"width\":"<<im.width<<",\"height\":"<<im.height<<"}";
            if (i+1<images.size()) j << ",";
            j << "\n";
        }
        j << "  ],\n  \"annotations\": [\n";
        for (size_t i=0;i<anns.size();++i){
            auto& a=anns[i];
            j << "    {\"id\":"<<a.id<<",\"image_id\":"<<a.image_id
              <<",\"category_id\":"<<a.category_id
              <<",\"bbox\":["<<a.x<<","<<a.y<<","<<a.w<<","<<a.h<<"]"
              <<",\"iscrowd\":0,\"area\":"<<a.area
              <<",\"instance_id\":"<<a.instance_id
              <<",\"label\":\""<<a.label<<"\"}";
            if (i+1<anns.size()) j << ",";
            j << "\n";
        }
        j << "  ],\n  \"categories\": [\n";
        for (size_t i=0;i<cats.size();++i){
            auto& c=cats[i];
            j << "    {\"id\":"<<c.id<<",\"name\":\""<<c.name<<"\"}";
            if (i+1<cats.size()) j << ",";
            j << "\n";
        }
        j << "  ]\n}\n";
    }
};
