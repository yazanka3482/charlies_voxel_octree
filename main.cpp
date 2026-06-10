#include "MeshTypes.hpp"
#include "MeshOctreeGen.hpp"
#include "VoxelOctree.hpp"
#include "AABB.hpp"
#include "CumulativeProfiler.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

namespace {

constexpr int kMeshLevel = 0;
constexpr int kMinOctreeDepth = 4;

void printUsage(const char* prog)
{
    std::cerr << "Usage: " << prog
              << " <input.stl> [-o output] [--cell-size SIZE] [-d DEPTH] [--cpu]\n"
              << "\n"
              << "  -o output       Output .voxel file path (default: output)\n"
              << "  --cell-size N   Target voxel size in world units (default: 0.01)\n"
              << "  -d DEPTH        Octree depth override (default: computed from cell size)\n"
              << "  --cpu           Force CPU voxelization (disable GPU even if built with OpenCL)\n";
}

int octreeDepthForCellSize(float maxDim, float targetCellSize)
{
    if (targetCellSize <= 0.f)
        targetCellSize = 0.01f;
    float ratio = std::max(maxDim / targetCellSize, 1.f);
    int depth = std::max(1, (int)std::ceil(std::log2(ratio)));
    return std::max(depth, kMinOctreeDepth);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath;
    std::string outputPath = "output";
    float cellSize = 0.01f;
    int depthOverride = -1;
    bool useGpu = true;

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            outputPath = argv[++i];
        }
        else if (std::strcmp(argv[i], "--cell-size") == 0 && i + 1 < argc)
        {
            cellSize = (float)std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc)
        {
            depthOverride = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--cpu") == 0)
        {
            useGpu = false;
        }
        else if (argv[i][0] == '-')
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
        else if (inputPath.empty())
        {
            inputPath = argv[i];
        }
        else
        {
            std::cerr << "Unexpected argument: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputPath.empty())
    {
        printUsage(argv[0]);
        return 1;
    }

    CumulativeProfiler::reset();

    objItem mesh;
    {
        PROFILE_SCOPE("stl/load_total");
        loadSTLFile(inputPath, mesh);
    }
    mesh.print();

    geo::aabb bounds = geo::getBoundingBoxOfPtCloud(mesh.positions);
    cgVec3 size = bounds.c2 - bounds.c1;
    float maxDim = std::max(size.x, std::max(size.y, size.z));
    if (maxDim < 1e-9f)
    {
        std::cerr << "Degenerate mesh bounds\n";
        return 1;
    }

    int depth = depthOverride >= 0 ? depthOverride : octreeDepthForCellSize(maxDim, cellSize);
    std::cout << "Voxelizing with cell size: " << cellSize << " depth: " << depth << "\n";
    std::cout << "Mesh bounds: "
              << bounds.c1.x << " " << bounds.c1.y << " " << bounds.c1.z << " "
              << bounds.c2.x << " " << bounds.c2.y << " " << bounds.c2.z << "\n";

    VoxelOctree octree(bounds, depth);
    {
        PROFILE_SCOPE("voxel/fromMesh_total");
        MeshOctreeGen gen(&mesh, {depth, useGpu});
        gen.addToWorld(&octree, kMeshLevel);
    }

    octree.saveToFile(outputPath);
    std::cout << "Wrote octree to " << outputPath << " ("
              << octree.getFileSize() << " bytes on disk, "
              << octree.getMemoryUsage() << " bytes allocated)\n";

    CumulativeProfiler::printReport("STL mesh octree voxelization");
    return 0;
}
