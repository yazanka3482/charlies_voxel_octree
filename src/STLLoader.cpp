#include "MeshTypes.hpp"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

bool isAsciiStl(std::ifstream& file)
{
    char header[6] = {};
    file.read(header, 5);
    if (!file)
        return false;
    file.seekg(0);
    std::string h(header, 5);
    for (char& c : h)
        c = (char)std::tolower((unsigned char)c);
    return h == "solid";
}

bool loadBinaryStl(const std::string& filename, objItem& target)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open STL file: " << filename
                  << " — " << std::strerror(errno) << "\n";
        return false;
    }

    char header[80];
    file.read(header, 80);
    if (!file)
        return false;

    uint32_t triCount = 0;
    file.read(reinterpret_cast<char*>(&triCount), sizeof(triCount));
    if (!file)
        return false;

    target.positions.clear();
    target.normals.clear();
    target.indices.clear();
    target.textCoords.clear();
    target.filename = filename;
    target.name = "stl_mesh";
    target.positions.reserve(triCount * 3);
    target.normals.reserve(triCount * 3);
    target.indices.reserve(triCount * 3);

    for (uint32_t t = 0; t < triCount; t++)
    {
        float normal[3];
        float verts[9];
        file.read(reinterpret_cast<char*>(normal), sizeof(normal));
        file.read(reinterpret_cast<char*>(verts), sizeof(verts));
        uint16_t attr = 0;
        file.read(reinterpret_cast<char*>(&attr), sizeof(attr));
        if (!file)
        {
            std::cerr << "Unexpected end of binary STL: " << filename << "\n";
            return false;
        }

        int base = (int)target.positions.size();
        cgVec3 n(normal[0], normal[1], normal[2]);
        for (int v = 0; v < 3; v++)
        {
            target.positions.push_back(cgVec3(verts[v * 3 + 0], verts[v * 3 + 1], verts[v * 3 + 2]));
            target.normals.push_back(n);
            target.indices.push_back(base + v);
        }
    }

    return !target.positions.empty();
}

bool loadAsciiStl(const std::string& filename, objItem& target)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Failed to open STL file: " << filename
                  << " — " << std::strerror(errno) << "\n";
        return false;
    }

    target.positions.clear();
    target.normals.clear();
    target.indices.clear();
    target.textCoords.clear();
    target.filename = filename;
    target.name = "stl_mesh";

    std::string line;
    cgVec3 currentNormal;
    std::vector<cgVec3> triVerts;

    auto flushTriangle = [&]() {
        if (triVerts.size() != 3)
            return;
        int base = (int)target.positions.size();
        for (int i = 0; i < 3; i++)
        {
            target.positions.push_back(triVerts[i]);
            target.normals.push_back(currentNormal);
            target.indices.push_back(base + i);
        }
        triVerts.clear();
    };

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token.empty())
            continue;

        for (char& c : token)
            c = (char)std::tolower((unsigned char)c);

        if (token == "solid")
        {
            std::string name;
            iss >> name;
            if (!name.empty())
                target.name = name;
        }
        else if (token == "facet")
        {
            std::string normalToken;
            iss >> normalToken;
            if (normalToken == "normal")
            {
                float nx, ny, nz;
                iss >> nx >> ny >> nz;
                currentNormal = cgVec3(nx, ny, nz);
            }
            triVerts.clear();
        }
        else if (token == "vertex")
        {
            float x, y, z;
            iss >> x >> y >> z;
            triVerts.push_back(cgVec3(x, y, z));
        }
        else if (token == "endfacet")
        {
            flushTriangle();
        }
        else if (token == "endsolid")
        {
            break;
        }
    }

    return !target.positions.empty();
}

void ensureVertexNormals(objItem& mesh)
{
    if (mesh.normals.size() == mesh.positions.size())
        return;

    mesh.normals.assign(mesh.positions.size(), cgVec3(0, 0, 0));
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
    {
        int i0 = mesh.indices[i];
        int i1 = mesh.indices[i + 1];
        int i2 = mesh.indices[i + 2];
        cgVec3 e1 = mesh.positions[i1] - mesh.positions[i0];
        cgVec3 e2 = mesh.positions[i2] - mesh.positions[i0];
        cgVec3 n = cross(e1, e2);
        float len = n.length();
        if (len > 1e-12f)
            n = n * (1.f / len);
        mesh.normals[i0] = mesh.normals[i0] + n;
        mesh.normals[i1] = mesh.normals[i1] + n;
        mesh.normals[i2] = mesh.normals[i2] + n;
    }

    for (size_t i = 0; i < mesh.normals.size(); i++)
    {
        float len = mesh.normals[i].length();
        if (len > 1e-12f)
            mesh.normals[i] = mesh.normals[i] * (1.f / len);
        else
            mesh.normals[i] = cgVec3(0, 0, 1);
    }
}

} // namespace

void loadSTLFile(const std::string& filename, objItem& target)
{
    std::ifstream probe(filename, std::ios::binary);
    if (!probe)
    {
        std::cerr << "Failed to open STL file: " << filename
                  << " — " << std::strerror(errno) << "\n";
        std::exit(1);
    }

    bool ok = false;
    if (isAsciiStl(probe))
        ok = loadAsciiStl(filename, target);
    else
        ok = loadBinaryStl(filename, target);

    if (!ok || target.positions.empty())
    {
        std::cerr << "Failed to load STL file: " << filename << "\n";
        std::exit(1);
    }

    ensureVertexNormals(target);
}
