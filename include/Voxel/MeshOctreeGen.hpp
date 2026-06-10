#ifndef MESHOCTREEGEN
#define MESHOCTREEGEN

#include "OctreeGeneration.hpp"
#include "MeshTypes.hpp"
#include "GeoPrimitives.hpp"
#include "TriangleBVH.hpp"
#include <memory>

struct MeshOctreeGenParams{
    int octreeDepth;
    bool useGpu = true;
};
/**
 * @brief Generates an octree from a mesh
 * 
 */
class MeshOctreeGen : public OctreeGenerater{
    public:
    
    MeshOctreeGen(objItem* mesh, MeshOctreeGenParams params);

    virtual ~MeshOctreeGen();

    /**
     * @brief Get the volume description for a voxel
     * 
     * @param voxel 
     * @return OctreeVoxelGenDesc 
     */
    OctreeVoxelGenDesc getVolumeDesc(geo::aabb voxel) override;
    void getVolumeDescBatch(const OctreeVoxelQuery* queries, OctreeVoxelGenDesc* out, size_t count) override;

    private:

    static std::vector<geo::Facet> buildMeshFacets(objItem* mesh);

    objItem* m_Mesh;
    TriangleBVH m_BVH;
    MeshOctreeGenParams m_Params;
#ifdef VOXEL_USE_OPENCL
    std::unique_ptr<class MeshOctreeGpu> m_GpuEvaluator;
#endif
    bool m_UseGpu = false;


};

#endif /* MESHOCTREEGEN */
