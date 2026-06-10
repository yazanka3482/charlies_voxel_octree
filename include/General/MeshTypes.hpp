#ifndef MESHTYPES_HPP
#define MESHTYPES_HPP

#include "cgVec.hpp"
#include <iostream>
#include <string>
#include <vector>

struct objItem {
    std::vector<cgVec3> positions;
    std::vector<cgVec3> normals;
    std::vector<cgVec2> textCoords;
    std::vector<int> indices;

    std::string matName;
    std::string name;
    std::string filename;
    std::string mtllib;

    void print() {
        std::cout << "Mesh: " << name << " (" << filename << ")\n";
        std::cout << "  vertices: " << positions.size() << "\n";
        std::cout << "  triangles: " << indices.size() / 3 << "\n";
    }
};

void loadSTLFile(const std::string& filename, objItem& target);

#endif
