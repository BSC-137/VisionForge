#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <tuple>


inline void write_yolo_txt(const std::string& path, int W,int H,
                           const std::vector<std::tuple<int,int,int,int,int>>& boxes)
{
    // tuple: (class_id, x0,y0,x1,y1)
    std::ofstream f(path);
    for (auto& t: boxes) {
        int cls,x0,y0,x1,y1; std::tie(cls,x0,y0,x1,y1)=t;
        double cx = (x0 + x1 + 1)/2.0, cy = (y0 + y1 + 1)/2.0;
        double w  = (x1 - x0 + 1),     h  = (y1 - y0 + 1);
        f << cls << " " << cx/W << " " << cy/H << " " << w/W << " " << h/H << "\n";
    }
}
