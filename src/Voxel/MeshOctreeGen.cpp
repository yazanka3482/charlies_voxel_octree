#include "MeshOctreeGen.hpp"
#include "CumulativeProfiler.hpp"
#ifdef VOXEL_USE_OPENCL
#include "MeshOctreeGpu.hpp"
#include "OpenCLContext.hpp"
#endif
#include <iostream>

MeshOctreeGen::~MeshOctreeGen() = default;

MeshOctreeGen::MeshOctreeGen(objItem* mesh, MeshOctreeGenParams params)
    : m_Mesh(mesh)
    , m_Params(params)
{
    std::vector<geo::Facet> facets;
    {
        PROFILE_SCOPE("voxel/build_facets");
        facets = buildMeshFacets(mesh);
    }
    {
        PROFILE_SCOPE("voxel/bvh_build");
        m_BVH = TriangleBVH(mesh->positions, facets);
    }

#ifdef VOXEL_USE_OPENCL
    if (m_Params.useGpu)
    {
        if (OpenCLContext::instance().isAvailable())
        {
            m_GpuEvaluator = std::make_unique<MeshOctreeGpu>(m_BVH);
            m_UseGpu = m_GpuEvaluator && m_GpuEvaluator->isReady();
        }
        if (!m_UseGpu)
            std::cerr << "MeshOctreeGen: GPU path unavailable, using CPU fallback." << std::endl;
    }
#else
    (void)m_Params.useGpu;
#endif
}

std::vector<geo::Facet> MeshOctreeGen::buildMeshFacets(objItem* mesh)
{
    std::vector<geo::Facet> facets;
    facets.reserve(mesh->indices.size() / 3);
    for (int i = 0; i < (int)mesh->indices.size(); i += 3)
        facets.push_back(geo::Facet(mesh->indices[i], mesh->indices[i + 1], mesh->indices[i + 2]));
    return facets;
}

OctreeVoxelGenDesc MeshOctreeGen::getVolumeDesc(geo::aabb voxel) {
    OctreeVoxelQuery query;
    query.volume = voxel;
    OctreeVoxelGenDesc out;
    getVolumeDescBatch(&query, &out, 1);
    return out;
}

void MeshOctreeGen::getVolumeDescBatch(const OctreeVoxelQuery* queries, OctreeVoxelGenDesc* out, size_t count)
{
#ifdef VOXEL_USE_OPENCL
    if (m_UseGpu && m_GpuEvaluator)
    {
        PROFILE_SCOPE("voxel/getVolumeDescBatch_gpu");
        for (size_t i = 0; i < count; i++)
            out[i] = OctreeVoxelGenDesc({outside, 0, false});
        std::vector<geo::aabb> voxels(count);
        for (size_t i = 0; i < count; i++)
            voxels[i] = queries[i].volume;
        m_GpuEvaluator->evaluateBatch(voxels.data(), out, count);
        return;
    }
#endif

    for (size_t i = 0; i < count; i++)
    {
        geo::aabb voxel = queries[i].volume;
        static const int kCorners = CumulativeProfiler::intern("voxel/getVolumeDesc_corners");
        Stopwatch sw;
        sw.start();
        voxel.calcAllCorners();
        CumulativeProfiler::addSample(kCorners, sw.stop_time());

        static const int kAabb = CumulativeProfiler::intern("voxel/getVolumeDesc_aabbIntersect");
        sw.start();
        bool hitsSurface = m_BVH.aabbIntersectsAnyTriangle(voxel);
        CumulativeProfiler::addSample(kAabb, sw.stop_time());
        if (hitsSurface)
        {
            out[i] = OctreeVoxelGenDesc({intersecting, 1, false});
            continue;
        }

        static const int kParity = CumulativeProfiler::intern("voxel/getVolumeDesc_parity");
        bool allInside = true;
        bool allOutside = true;
        for(int c = 0; c < 8; c++){
            sw.start();
            bool inside = m_BVH.containsPointParity(voxel.allCorners[c]);
            CumulativeProfiler::addSample(kParity, sw.stop_time());
            if(inside){
                allOutside = false;
            } else {
                allInside = false;
            }
        }
        if(allInside){
            out[i] = OctreeVoxelGenDesc({inside, 1, false});
        } else if(allOutside){
            out[i] = OctreeVoxelGenDesc({outside, 0, false});
        } else {
            out[i] = OctreeVoxelGenDesc({intersecting, 1, false});
        }
    }
}
