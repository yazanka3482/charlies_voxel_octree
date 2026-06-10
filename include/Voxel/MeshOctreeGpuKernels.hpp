#ifndef MESH_OCTREE_GPU_KERNELS
#define MESH_OCTREE_GPU_KERNELS

// OpenCL kernel source for batched mesh octree voxel classification.
static const char kMeshOctreeGpuKernelSource[] = R"CL(
typedef struct {
    float lmin[3];
    uint left;
    float lmax[3];
    uint right;
    float rmin[3];
    uint triCount;
    float rmax[3];
    uint firstTri;
} BvhNode;

typedef struct {
    float c1[3];
    float pad0;
    float c2[3];
    float pad1;
} VoxelAabb;

typedef struct {
    uint loc;
    uint type;
    uint noneType;
    uint pad;
} VoxelResult;

#define LOC_INSIDE 0
#define LOC_OUTSIDE 1
#define LOC_INTERSECTING 2

bool boxOverlap(float3 amin, float3 amax, float3 bmin, float3 bmax)
{
    return !(amin.x > bmax.x || amax.x < bmin.x ||
             amin.y > bmax.y || amax.y < bmin.y ||
             amin.z > bmax.z || amax.z < bmin.z);
}

bool axisOverlap(float p0, float p1, float rad)
{
    float mn = fmin(p0, p1);
    float mx = fmax(p0, p1);
    return !(mn > rad || mx < -rad);
}

bool triangleOverlapsAabb(float3 qc1, float3 qc2, float3 v0, float3 v1, float3 v2)
{
    float3 center = (qc1 + qc2) * 0.5f;
    float ex = (qc2.x - qc1.x) * 0.5f;
    float ey = (qc2.y - qc1.y) * 0.5f;
    float ez = (qc2.z - qc1.z) * 0.5f;

    float3 tv0 = v0 - center;
    float3 tv1 = v1 - center;
    float3 tv2 = v2 - center;

    float3 e0 = tv1 - tv0;
    float3 e1 = tv2 - tv1;
    float3 e2 = tv0 - tv2;

    float minv = fmin(tv0.x, fmin(tv1.x, tv2.x));
    float maxv = fmax(tv0.x, fmax(tv1.x, tv2.x));
    if (minv > ex || maxv < -ex) return false;

    minv = fmin(tv0.y, fmin(tv1.y, tv2.y));
    maxv = fmax(tv0.y, fmax(tv1.y, tv2.y));
    if (minv > ey || maxv < -ey) return false;

    minv = fmin(tv0.z, fmin(tv1.z, tv2.z));
    maxv = fmax(tv0.z, fmax(tv1.z, tv2.z));
    if (minv > ez || maxv < -ez) return false;

    float3 normal = cross(e0, e1);
    float d = dot(normal, tv0);
    float r = ex * fabs(normal.x) + ey * fabs(normal.y) + ez * fabs(normal.z);
    if (fabs(d) > r) return false;

    float rad;

    rad = ey * fabs(e0.z) + ez * fabs(e0.y);
    if (!axisOverlap(e0.z * tv0.y - e0.y * tv0.z, e0.z * tv2.y - e0.y * tv2.z, rad)) return false;
    rad = ez * fabs(e0.x) + ex * fabs(e0.z);
    if (!axisOverlap(e0.x * tv0.z - e0.z * tv0.x, e0.x * tv2.z - e0.z * tv2.x, rad)) return false;
    rad = ex * fabs(e0.y) + ey * fabs(e0.x);
    if (!axisOverlap(e0.y * tv1.x - e0.x * tv1.y, e0.y * tv2.x - e0.x * tv2.y, rad)) return false;

    rad = ey * fabs(e1.z) + ez * fabs(e1.y);
    if (!axisOverlap(e1.z * tv0.y - e1.y * tv0.z, e1.z * tv2.y - e1.y * tv2.z, rad)) return false;
    rad = ez * fabs(e1.x) + ex * fabs(e1.z);
    if (!axisOverlap(e1.x * tv0.z - e1.z * tv0.x, e1.x * tv2.z - e1.z * tv2.x, rad)) return false;
    rad = ex * fabs(e1.y) + ey * fabs(e1.x);
    if (!axisOverlap(e1.y * tv0.x - e1.x * tv0.y, e1.y * tv1.x - e1.x * tv1.y, rad)) return false;

    rad = ey * fabs(e2.z) + ez * fabs(e2.y);
    if (!axisOverlap(e2.z * tv0.y - e2.y * tv0.z, e2.z * tv1.y - e2.y * tv1.z, rad)) return false;
    rad = ez * fabs(e2.x) + ex * fabs(e2.z);
    if (!axisOverlap(e2.x * tv0.z - e2.z * tv0.x, e2.x * tv1.z - e2.z * tv1.x, rad)) return false;
    rad = ex * fabs(e2.y) + ey * fabs(e2.x);
    if (!axisOverlap(e2.y * tv1.x - e2.x * tv1.y, e2.y * tv2.x - e2.x * tv2.y, rad)) return false;

    return true;
}

void fetchTriangle(
    uint primListIdx,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled,
    float3* v0,
    float3* v1,
    float3* v2)
{
    uint prim = primIdx[primListIdx];
    uint vidx = prim * 3;
    uint i0, i1, i2;
    if (indexedEnabled != 0)
    {
        i0 = vertIdx[vidx];
        i1 = vertIdx[vidx + 1];
        i2 = vertIdx[vidx + 2];
    }
    else
    {
        i0 = vidx;
        i1 = vidx + 1;
        i2 = vidx + 2;
    }
    float4 a = vertices[i0];
    float4 b = vertices[i1];
    float4 c = vertices[i2];
    *v0 = (float3)(a.x, a.y, a.z);
    *v1 = (float3)(b.x, b.y, b.z);
    *v2 = (float3)(c.x, c.y, c.z);
}

// Iterative traversal: Apple's cl2Metal translator does not support recursion.
bool aabbIntersectsAnyTriangleGpu(
    float3 qc1, float3 qc2,
    __global const BvhNode* nodes,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled)
{
    uint stack[64];
    uint stackPtr = 0;
    uint cur = 0;

    while (true)
    {
        BvhNode node = nodes[cur];
        if (node.triCount > 0)
        {
            for (uint i = 0; i < node.triCount; i++)
            {
                float3 v0, v1, v2;
                fetchTriangle(node.firstTri + i, vertices, primIdx, vertIdx, indexedEnabled, &v0, &v1, &v2);
                if (triangleOverlapsAabb(qc1, qc2, v0, v1, v2))
                    return true;
            }
            if (stackPtr == 0)
                break;
            cur = stack[--stackPtr];
            continue;
        }

        float3 lmin = (float3)(node.lmin[0], node.lmin[1], node.lmin[2]);
        float3 lmax = (float3)(node.lmax[0], node.lmax[1], node.lmax[2]);
        float3 rmin = (float3)(node.rmin[0], node.rmin[1], node.rmin[2]);
        float3 rmax = (float3)(node.rmax[0], node.rmax[1], node.rmax[2]);

        int visitLeft = boxOverlap(lmin, lmax, qc1, qc2);
        int visitRight = boxOverlap(rmin, rmax, qc1, qc2);

        if (visitLeft && visitRight)
        {
            stack[stackPtr++] = node.right;
            cur = node.left;
        }
        else if (visitLeft)
        {
            cur = node.left;
        }
        else if (visitRight)
        {
            cur = node.right;
        }
        else
        {
            if (stackPtr == 0)
                break;
            cur = stack[--stackPtr];
        }
    }
    return false;
}

bool rayIntersectsTriangleParity(float3 rayOrigin, float3 rayVector, float3 tri0, float3 tri1, float3 tri2, float tMin)
{
    const float EPSILON = 1e-7f;
    const float BARY_EPS = 1e-5f;
    float3 edge1 = tri1 - tri0;
    float3 edge2 = tri2 - tri0;
    float3 h = cross(rayVector, edge2);
    float a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;
    float f = 1.0f / a;
    float3 s = rayOrigin - tri0;
    float u = f * dot(s, h);
    if (u < -BARY_EPS || u > 1.0f + BARY_EPS)
        return false;
    float3 q = cross(s, edge1);
    float v = f * dot(rayVector, q);
    if (v < -BARY_EPS || u + v > 1.0f + BARY_EPS)
        return false;
    float t = f * dot(edge2, q);
    return t > tMin;
}

int countHitsGpu(
    float3 O, float3 D, float3 rD, float tMin, uint nodeIdx,
    __global const BvhNode* nodes,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled)
{
    const float BVH_FAR = 1e30f;
    int hits = 0;
    uint stack[64];
    uint stackPtr = 0;
    uint cur = nodeIdx;

    while (true)
    {
        BvhNode node = nodes[cur];
        if (node.triCount > 0)
        {
            for (uint i = 0; i < node.triCount; i++)
            {
                float3 v0, v1, v2;
                fetchTriangle(node.firstTri + i, vertices, primIdx, vertIdx, indexedEnabled, &v0, &v1, &v2);
                if (rayIntersectsTriangleParity(O, D, v0, v1, v2, tMin))
                    hits++;
            }
            if (stackPtr == 0)
                break;
            cur = stack[--stackPtr];
            continue;
        }

        float3 lmin = (float3)(node.lmin[0], node.lmin[1], node.lmin[2]) - O;
        float3 lmax = (float3)(node.lmax[0], node.lmax[1], node.lmax[2]) - O;
        float3 rmin = (float3)(node.rmin[0], node.rmin[1], node.rmin[2]) - O;
        float3 rmax = (float3)(node.rmax[0], node.rmax[1], node.rmax[2]) - O;

        float3 t1a = lmin * rD;
        float3 t2a = lmax * rD;
        float3 t1b = rmin * rD;
        float3 t2b = rmax * rD;

        float tmina = fmax(fmax(fmin(t1a.x, t2a.x), fmin(t1a.y, t2a.y)), fmin(t1a.z, t2a.z));
        float tmaxa = fmin(fmin(fmax(t1a.x, t2a.x), fmax(t1a.y, t2a.y)), fmax(t1a.z, t2a.z));
        float tminb = fmax(fmax(fmin(t1b.x, t2b.x), fmin(t1b.y, t2b.y)), fmin(t1b.z, t2b.z));
        float tmaxb = fmin(fmin(fmax(t1b.x, t2b.x), fmax(t1b.y, t2b.y)), fmax(t1b.z, t2b.z));

        float dist1 = BVH_FAR;
        float dist2 = BVH_FAR;
        if (tmaxa >= tmina && tmaxa >= 0.0f)
            dist1 = tmina;
        if (tmaxb >= tminb && tmaxb >= 0.0f)
            dist2 = tminb;

        uint lidx = node.left;
        uint ridx = node.right;
        if (dist1 > dist2)
        {
            float t = dist1; dist1 = dist2; dist2 = t;
            uint tmp = lidx; lidx = ridx; ridx = tmp;
        }

        if (dist1 == BVH_FAR)
        {
            if (stackPtr == 0)
                break;
            cur = stack[--stackPtr];
        }
        else
        {
            cur = lidx;
            if (dist2 != BVH_FAR)
                stack[stackPtr++] = ridx;
        }
    }
    return hits;
}

bool containsPointParityGpu(
    float3 point, float parityBias,
    __global const BvhNode* nodes,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled,
    uint triCount)
{
    if (triCount == 0)
        return false;

    float3 dir = (float3)(0.3407109204f, 0.7575350765f, -0.5568273310f);
    float invLen = native_rsqrt(dot(dir, dir));
    dir *= invLen;

    float tMin = parityBias * 0.5f;
    float3 origin = point + dir * parityBias;
    float3 rD = (float3)(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int hits = countHitsGpu(origin, dir, rD, tMin, 0, nodes, vertices, primIdx, vertIdx, indexedEnabled);
    return (hits & 1) == 1;
}

void classifyVoxel(
    VoxelAabb voxel,
    float parityBias,
    __global const BvhNode* nodes,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled,
    uint triCount,
    __global VoxelResult* outResult)
{
    float3 qc1 = (float3)(voxel.c1[0], voxel.c1[1], voxel.c1[2]);
    float3 qc2 = (float3)(voxel.c2[0], voxel.c2[1], voxel.c2[2]);

    if (triCount > 0 && aabbIntersectsAnyTriangleGpu(qc1, qc2, nodes, vertices, primIdx, vertIdx, indexedEnabled))
    {
        outResult->loc = LOC_INTERSECTING;
        outResult->type = 1;
        outResult->noneType = 0;
        return;
    }

    float corners[8][3] = {
        {voxel.c1[0], voxel.c1[1], voxel.c1[2]},
        {voxel.c2[0], voxel.c1[1], voxel.c1[2]},
        {voxel.c1[0], voxel.c2[1], voxel.c1[2]},
        {voxel.c2[0], voxel.c2[1], voxel.c1[2]},
        {voxel.c1[0], voxel.c1[1], voxel.c2[2]},
        {voxel.c2[0], voxel.c1[1], voxel.c2[2]},
        {voxel.c1[0], voxel.c2[1], voxel.c2[2]},
        {voxel.c2[0], voxel.c2[1], voxel.c2[2]}
    };

    int allInside = 1;
    int allOutside = 1;
    for (int i = 0; i < 8; i++)
    {
        float3 p = (float3)(corners[i][0], corners[i][1], corners[i][2]);
        int inside = containsPointParityGpu(p, parityBias, nodes, vertices, primIdx, vertIdx, indexedEnabled, triCount);
        if (inside)
            allOutside = 0;
        else
            allInside = 0;
    }

    if (allInside)
    {
        outResult->loc = LOC_INSIDE;
        outResult->type = 1;
        outResult->noneType = 0;
        return;
    }
    if (allOutside)
    {
        outResult->loc = LOC_OUTSIDE;
        outResult->type = 0;
        outResult->noneType = 0;
        return;
    }
    outResult->loc = LOC_INTERSECTING;
    outResult->type = 1;
    outResult->noneType = 0;
}

__kernel void classifyVoxels(
    __global const VoxelAabb* voxels,
    __global VoxelResult* results,
    uint count,
    float parityBias,
    __global const BvhNode* nodes,
    __global const float4* vertices,
    __global const uint* primIdx,
    __global const uint* vertIdx,
    uint indexedEnabled,
    uint triCount)
{
    uint gid = get_global_id(0);
    if (gid >= count)
        return;
    classifyVoxel(voxels[gid], parityBias, nodes, vertices, primIdx, vertIdx, indexedEnabled, triCount, &results[gid]);
}
)CL";

#endif
