#ifndef TRIANGLE_BVH
#define TRIANGLE_BVH

#include "GeoPrimitives.hpp"
#include "tiny_bvh.h"
#include <cstdint>
#include <vector>

class TriangleBVH {
public:
#ifdef VOXEL_USE_OPENCL
    struct GpuBvhNode {
        float lmin[3];
        uint32_t left;
        float lmax[3];
        uint32_t right;
        float rmin[3];
        uint32_t triCount;
        float rmax[3];
        uint32_t firstTri;
    };

    struct GpuLayout {
        std::vector<GpuBvhNode> nodes;
        std::vector<tinybvh::bvhvec4> vertices;
        std::vector<uint32_t> primIdx;
        std::vector<uint32_t> vertIdx;
        uint32_t triCount = 0;
        uint32_t nodeCount = 0;
        bool indexed = false;
        float parityRayBias = 0.f;
    };
#endif

    TriangleBVH() = default;
    TriangleBVH(const std::vector<cgVec3>& pts, const std::vector<geo::Facet>& facets);
    TriangleBVH(const TriangleBVH&) = delete;
    TriangleBVH& operator=(const TriangleBVH&) = delete;
    TriangleBVH(TriangleBVH&& other) noexcept;
    TriangleBVH& operator=(TriangleBVH&& other) noexcept;
    ~TriangleBVH() = default;

    bool containsPointParity(cgVec3 point) const;

    /** True if any triangle in the BVH overlaps the AABB. */
    bool aabbIntersectsAnyTriangle(const geo::aabb& box) const;

    int nodeCount() const;
    int rootIndex() const { return m_BVH.bvhNode ? 0 : -1; }
    geo::aabb rootBounds() const;
#ifdef VOXEL_USE_OPENCL
    const GpuLayout& gpuLayout() const { return m_GpuLayout; }
#endif

private:
    std::vector<tinybvh::bvhvec4> m_Vertices;
    std::vector<uint32_t> m_Indices;
    tinybvh::BVH m_BVH;
#ifdef VOXEL_USE_OPENCL
    tinybvh::BVH_GPU m_BvhGpu;
    GpuLayout m_GpuLayout;

    void buildGpuLayout();
#endif

    void getTriangle(uint32_t primListIdx, cgVec3& v0, cgVec3& v1, cgVec3& v2) const;
    float parityRayBias() const;
    bool rayHitsAabb(const geo::aabb& b, cgVec3 origin, cgVec3 dir) const;
    void countHits(cgVec3 origin, cgVec3 dir, float tMin, uint32_t nodeIdx, int& hits) const;
    bool aabbIntersectsAnyTriangleAtNode(const geo::aabb& box, uint32_t nodeIdx) const;
};

#endif
