#define TINYBVH_IMPLEMENTATION
#include "TriangleBVH.hpp"
#include "CumulativeProfiler.hpp"
#include "Stopwatch.hpp"
#include <algorithm>
#include <cmath>

namespace {

// SAT interval overlap: triangle projection [mn,mx] vs box [-rad,+rad] on one axis.
bool axisOverlap(float p0, float p1, float rad)
{
    float mn = std::min(p0, p1);
    float mx = std::max(p0, p1);
    return !(mn > rad || mx < -rad);
}

// Akenine-Moller triangle-AABB overlap (box center at origin, half-extents ex,ey,ez)
bool triangleOverlapsAabb(const geo::aabb& box, const cgVec3& v0, const cgVec3& v1, const cgVec3& v2)
{
    cgVec3 center(
        (box.c1.x + box.c2.x) * 0.5f,
        (box.c1.y + box.c2.y) * 0.5f,
        (box.c1.z + box.c2.z) * 0.5f);
    float ex = (box.c2.x - box.c1.x) * 0.5f;
    float ey = (box.c2.y - box.c1.y) * 0.5f;
    float ez = (box.c2.z - box.c1.z) * 0.5f;

    cgVec3 tv0 = cgVec3(v0.x - center.x, v0.y - center.y, v0.z - center.z);
    cgVec3 tv1 = cgVec3(v1.x - center.x, v1.y - center.y, v1.z - center.z);
    cgVec3 tv2 = cgVec3(v2.x - center.x, v2.y - center.y, v2.z - center.z);

    cgVec3 e0 = tv1 - tv0;
    cgVec3 e1 = tv2 - tv1;
    cgVec3 e2 = tv0 - tv2;

    float minv, maxv;

    minv = std::min(tv0.x, std::min(tv1.x, tv2.x));
    maxv = std::max(tv0.x, std::max(tv1.x, tv2.x));
    if (minv > ex || maxv < -ex)
        return false;

    minv = std::min(tv0.y, std::min(tv1.y, tv2.y));
    maxv = std::max(tv0.y, std::max(tv1.y, tv2.y));
    if (minv > ey || maxv < -ey)
        return false;

    minv = std::min(tv0.z, std::min(tv1.z, tv2.z));
    maxv = std::max(tv0.z, std::max(tv1.z, tv2.z));
    if (minv > ez || maxv < -ez)
        return false;

    cgVec3 normal = cross(e0, e1);
    float d = dot(normal, tv0);
    float r = ex * std::fabs(normal.x) + ey * std::fabs(normal.y) + ez * std::fabs(normal.z);
    if (std::fabs(d) > r)
        return false;

    // Nine edge-axis tests (vertex pairs per Tomas Akenine-Moller tribox2).
    float rad, p0, p1;

    // e0 = tv1 - tv0
    rad = ey * std::fabs(e0.z) + ez * std::fabs(e0.y);
    if (!axisOverlap(e0.z * tv0.y - e0.y * tv0.z, e0.z * tv2.y - e0.y * tv2.z, rad))
        return false;
    rad = ez * std::fabs(e0.x) + ex * std::fabs(e0.z);
    if (!axisOverlap(e0.x * tv0.z - e0.z * tv0.x, e0.x * tv2.z - e0.z * tv2.x, rad))
        return false;
    rad = ex * std::fabs(e0.y) + ey * std::fabs(e0.x);
    if (!axisOverlap(e0.y * tv1.x - e0.x * tv1.y, e0.y * tv2.x - e0.x * tv2.y, rad))
        return false;

    // e1 = tv2 - tv1
    rad = ey * std::fabs(e1.z) + ez * std::fabs(e1.y);
    if (!axisOverlap(e1.z * tv0.y - e1.y * tv0.z, e1.z * tv2.y - e1.y * tv2.z, rad))
        return false;
    rad = ez * std::fabs(e1.x) + ex * std::fabs(e1.z);
    if (!axisOverlap(e1.x * tv0.z - e1.z * tv0.x, e1.x * tv2.z - e1.z * tv2.x, rad))
        return false;
    rad = ex * std::fabs(e1.y) + ey * std::fabs(e1.x);
    if (!axisOverlap(e1.y * tv0.x - e1.x * tv0.y, e1.y * tv1.x - e1.x * tv1.y, rad))
        return false;

    // e2 = tv0 - tv2
    rad = ey * std::fabs(e2.z) + ez * std::fabs(e2.y);
    if (!axisOverlap(e2.z * tv0.y - e2.y * tv0.z, e2.z * tv1.y - e2.y * tv1.z, rad))
        return false;
    rad = ez * std::fabs(e2.x) + ex * std::fabs(e2.z);
    if (!axisOverlap(e2.x * tv0.z - e2.z * tv0.x, e2.x * tv1.z - e2.z * tv1.z, rad))
        return false;
    rad = ex * std::fabs(e2.y) + ey * std::fabs(e2.x);
    if (!axisOverlap(e2.y * tv1.x - e2.x * tv1.y, e2.y * tv2.x - e2.x * tv2.y, rad))
        return false;

    return true;
}

tinybvh::bvhvec3 toBvh(const cgVec3& v)
{
    return tinybvh::bvhvec3(v.x, v.y, v.z);
}

// Slab method: true if the forward ray segment [0, maxT] intersects the AABB.
bool rayIntersectsAabbForward(const geo::aabb& b, cgVec3 o, cgVec3 d, float maxT)
{
    float tmin = 0.f;
    float tmax = maxT;
    const float parallelEps = 1e-8f;
    const float slabEps = 1e-6f;

    auto slab = [&](float oa, float da, float bmin, float bmax) -> bool {
        if (std::fabs(da) < parallelEps)
            return oa >= bmin - slabEps && oa <= bmax + slabEps;
        float invD = 1.f / da;
        float t0 = (bmin - oa) * invD;
        float t1 = (bmax - oa) * invD;
        if (t0 > t1)
            std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        return tmax >= tmin;
    };

    if (!slab(o.x, d.x, b.c1.x, b.c2.x))
        return false;
    if (!slab(o.y, d.y, b.c1.y, b.c2.y))
        return false;
    if (!slab(o.z, d.z, b.c1.z, b.c2.z))
        return false;
    return tmax >= std::max(tmin, 0.f);
}

} // namespace

TriangleBVH::TriangleBVH(TriangleBVH&& other) noexcept
    : m_Vertices(std::move(other.m_Vertices))
    , m_Indices(std::move(other.m_Indices))
    , m_BVH(std::move(other.m_BVH))
#ifdef VOXEL_USE_OPENCL
    , m_BvhGpu(std::move(other.m_BvhGpu))
    , m_GpuLayout(std::move(other.m_GpuLayout))
#endif
{
}

TriangleBVH& TriangleBVH::operator=(TriangleBVH&& other) noexcept
{
    if (this != &other)
    {
        // tinybvh::BVH only has a move ctor (no move assign); default assign shallow-copies
        // heap pointers and causes double-free when the source is destroyed.
        m_BVH.~BVH();
        new (&m_BVH) tinybvh::BVH(std::move(other.m_BVH));
        m_Vertices = std::move(other.m_Vertices);
        m_Indices = std::move(other.m_Indices);
#ifdef VOXEL_USE_OPENCL
        m_BvhGpu.~BVH_GPU();
        new (&m_BvhGpu) tinybvh::BVH_GPU(std::move(other.m_BvhGpu));
        m_GpuLayout = std::move(other.m_GpuLayout);
#endif
    }
    return *this;
}

TriangleBVH::TriangleBVH(const std::vector<cgVec3>& pts, const std::vector<geo::Facet>& facets)
{
    PROFILE_SCOPE("bvh/copy_vertices_indices");
    m_Vertices.resize(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        m_Vertices[i] = tinybvh::bvhvec4(pts[i].x, pts[i].y, pts[i].z, 0.f);

    m_Indices.reserve(facets.size() * 3);
    for (const geo::Facet& f : facets)
    {
        m_Indices.push_back((uint32_t)f.inds[0]);
        m_Indices.push_back((uint32_t)f.inds[1]);
        m_Indices.push_back((uint32_t)f.inds[2]);
    }

    if (!facets.empty())
    {
        PROFILE_SCOPE("bvh/tinybvh_build");
        m_BVH.Build(m_Vertices.data(), m_Indices.data(), (uint32_t)facets.size());
#ifdef VOXEL_USE_OPENCL
        {
            PROFILE_SCOPE("bvh/tinybvh_gpu_convert");
            m_BvhGpu.ConvertFrom(m_BVH);
            buildGpuLayout();
        }
#endif
    }
}

#ifdef VOXEL_USE_OPENCL
void TriangleBVH::buildGpuLayout()
{
    m_GpuLayout = {};
    if (!m_BvhGpu.bvhNode || m_BvhGpu.triCount == 0)
        return;

    m_GpuLayout.nodeCount = m_BvhGpu.usedNodes;
    m_GpuLayout.triCount = m_BvhGpu.triCount;
    m_GpuLayout.parityRayBias = parityRayBias();
    m_GpuLayout.vertices = m_Vertices;

    m_GpuLayout.nodes.resize(m_GpuLayout.nodeCount);
    for (uint32_t i = 0; i < m_GpuLayout.nodeCount; i++)
    {
        const tinybvh::BVH_GPU::BVHNode& src = m_BvhGpu.bvhNode[i];
        GpuBvhNode& dst = m_GpuLayout.nodes[i];
        dst.lmin[0] = src.lmin.x; dst.lmin[1] = src.lmin.y; dst.lmin[2] = src.lmin.z;
        dst.lmax[0] = src.lmax.x; dst.lmax[1] = src.lmax.y; dst.lmax[2] = src.lmax.z;
        dst.rmin[0] = src.rmin.x; dst.rmin[1] = src.rmin.y; dst.rmin[2] = src.rmin.z;
        dst.rmax[0] = src.rmax.x; dst.rmax[1] = src.rmax.y; dst.rmax[2] = src.rmax.z;
        dst.left = src.left;
        dst.right = src.right;
        dst.triCount = src.triCount;
        dst.firstTri = src.firstTri;
    }

    const uint32_t primCount = m_BvhGpu.bvh.idxCount ? m_BvhGpu.bvh.idxCount : m_BvhGpu.triCount;
    m_GpuLayout.primIdx.resize(primCount);
    for (uint32_t i = 0; i < primCount; i++)
        m_GpuLayout.primIdx[i] = m_BvhGpu.bvh.primIdx[i];

    m_GpuLayout.indexed = m_BvhGpu.bvh.vertIdx != nullptr;
    if (m_GpuLayout.indexed)
    {
        m_GpuLayout.vertIdx.resize(primCount * 3);
        for (uint32_t i = 0; i < primCount * 3; i++)
            m_GpuLayout.vertIdx[i] = m_BvhGpu.bvh.vertIdx[i];
    }
}
#endif

void TriangleBVH::getTriangle(uint32_t primListIdx, cgVec3& v0, cgVec3& v1, cgVec3& v2) const
{
    const uint32_t prim = m_BVH.primIdx[primListIdx];
    const uint32_t vidx = prim * 3;
    auto corner = [&](int i) -> cgVec3 {
        const uint32_t vi = m_BVH.vertIdx ? m_BVH.vertIdx[vidx + i] : vidx + i;
        const tinybvh::bvhvec4& v = m_Vertices[vi];
        return cgVec3(v.x, v.y, v.z);
    };
    v0 = corner(0);
    v1 = corner(1);
    v2 = corner(2);
}

int TriangleBVH::nodeCount() const
{
    return m_BVH.bvhNode ? (int)m_BVH.NodeCount() : 0;
}

geo::aabb TriangleBVH::rootBounds() const
{
    if (!m_BVH.bvhNode || m_BVH.triCount == 0)
        return geo::aabb();
    return geo::aabb(
        cgVec3(m_BVH.aabbMin.x, m_BVH.aabbMin.y, m_BVH.aabbMin.z),
        cgVec3(m_BVH.aabbMax.x, m_BVH.aabbMax.y, m_BVH.aabbMax.z));
}

float TriangleBVH::parityRayBias() const
{
    if (!m_BVH.bvhNode || m_BVH.triCount == 0)
        return 1e-6f;
    const geo::aabb b = rootBounds();
    cgVec3 ext = b.c2 - b.c1;
    float scale = std::max(ext.x, std::max(ext.y, ext.z));
    return std::max(1e-7f, scale * 1e-5f);
}

bool TriangleBVH::rayHitsAabb(const geo::aabb& b, cgVec3 origin, cgVec3 dir) const
{
    return rayIntersectsAabbForward(b, origin, dir, 1e30f);
}

void TriangleBVH::countHits(cgVec3 origin, cgVec3 dir, float tMin, uint32_t nodeIdx, int& hits) const
{
    const tinybvh::BVH::BVHNode& n = m_BVH.bvhNode[nodeIdx];
    geo::aabb nodeBox(
        cgVec3(n.aabbMin.x, n.aabbMin.y, n.aabbMin.z),
        cgVec3(n.aabbMax.x, n.aabbMax.y, n.aabbMax.z));
    if (!rayHitsAabb(nodeBox, origin, dir))
        return;

    if (n.isLeaf())
    {
        cgVec3 hit;
        for (uint32_t i = 0; i < n.triCount; i++)
        {
            cgVec3 v0, v1, v2;
            getTriangle(n.leftFirst + i, v0, v1, v2);
            if (geo::rayIntersectsTriangleParity(origin, dir, v0, v1, v2, tMin, hit))
                hits++;
        }
        return;
    }

    countHits(origin, dir, tMin, n.leftFirst, hits);
    countHits(origin, dir, tMin, n.leftFirst + 1, hits);
}

bool TriangleBVH::containsPointParity(cgVec3 point) const
{
    if (!m_BVH.bvhNode || m_BVH.triCount == 0)
        return false;

    // A single parity ray is fooled when it crosses the surface exactly on a
    // shared triangle edge/vertex: the lenient barycentric tolerance in
    // rayIntersectsTriangleParity lets every triangle adjacent to that feature
    // register the one crossing, so it gets counted with the wrong multiplicity
    // and the parity flips (spurious inside/outside). The degeneracy depends on
    // the ray direction, so we vote across several incoherent directions; a
    // feature can only fool one of them at a time.
    static const cgVec3 dirs[] = {
        { 0.3407109204f,  0.7575350765f, -0.5568273310f},
    };
    constexpr int kRays = (int)(sizeof(dirs) / sizeof(dirs[0]));

    const float bias = parityRayBias();
    const float tMin = bias * 0.5f;

    int insideVotes = 0;
    for (int r = 0; r < kRays; r++)
    {
        // Nudge the origin per-ray so we never start exactly on the surface and
        // so collinear grid alignments don't repeat across directions.
        cgVec3 origin(
            point.x + bias * dirs[r].x,
            point.y + bias * dirs[r].y,
            point.z + bias * dirs[r].z);
        int hits = 0;
        countHits(origin, dirs[r], tMin, 0, hits);
        if ((hits & 1) == 1)
            insideVotes++;
    }
    return insideVotes * 2 > kRays;
}

bool TriangleBVH::aabbIntersectsAnyTriangleAtNode(const geo::aabb& box, uint32_t nodeIdx) const
{
    const tinybvh::BVH::BVHNode& n = m_BVH.bvhNode[nodeIdx];
    if (!n.Intersect(toBvh(box.c1), toBvh(box.c2)))
        return false;

    if (n.isLeaf())
    {
        for (uint32_t i = 0; i < n.triCount; i++)
        {
            cgVec3 v0, v1, v2;
            getTriangle(n.leftFirst + i, v0, v1, v2);
            if (triangleOverlapsAabb(box, v0, v1, v2))
                return true;
        }
        return false;
    }

    return aabbIntersectsAnyTriangleAtNode(box, n.leftFirst)
        || aabbIntersectsAnyTriangleAtNode(box, n.leftFirst + 1);
}

bool TriangleBVH::aabbIntersectsAnyTriangle(const geo::aabb& box) const
{
    if (!m_BVH.bvhNode || m_BVH.triCount == 0)
        return false;
    return aabbIntersectsAnyTriangleAtNode(box, 0);
}
