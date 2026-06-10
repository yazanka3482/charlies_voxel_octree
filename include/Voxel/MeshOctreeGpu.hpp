#ifndef MESH_OCTREE_GPU
#define MESH_OCTREE_GPU

#include "AABB.hpp"
#include "OctreeGeneration.hpp"
#include "OpenCLContext.hpp"
#include "TriangleBVH.hpp"

/** OpenCL batch evaluator for mesh octree voxel classification. */
class MeshOctreeGpu {
public:
    explicit MeshOctreeGpu(const TriangleBVH& bvh);
    ~MeshOctreeGpu();

    MeshOctreeGpu(const MeshOctreeGpu&) = delete;
    MeshOctreeGpu& operator=(const MeshOctreeGpu&) = delete;

    bool isReady() const { return m_Ready; }

    void evaluateBatch(const geo::aabb* voxels, OctreeVoxelGenDesc* out, size_t count);

private:
    struct GpuVoxelAabb {
        float c1[3];
        float pad0;
        float c2[3];
        float pad1;
    };

    struct GpuVoxelResult {
        uint32_t loc;
        uint32_t type;
        uint32_t noneType;
        uint32_t pad;
    };

    bool m_Ready = false;
    cl_program m_Program = nullptr;
    cl_kernel m_Kernel = nullptr;
    cl_mem m_BvhNodes = nullptr;
    cl_mem m_Vertices = nullptr;
    cl_mem m_PrimIdx = nullptr;
    cl_mem m_VertIdx = nullptr;
    cl_mem m_VoxelIn = nullptr;
    cl_mem m_ResultOut = nullptr;

    size_t m_VoxelBufferCapacity = 0;
    uint32_t m_TriCount = 0;
    uint32_t m_IndexedEnabled = 0;
    float m_ParityBias = 0.f;
};

#endif
