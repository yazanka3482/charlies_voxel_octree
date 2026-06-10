#include "AABB.hpp"
#include <cmath>
#include <iostream>

namespace
{
constexpr float MIN_DIST = 0.0001f;

cgVec3 linePlaneIntersectionPt(cgVec3 l_orig, cgVec3 l_dir, cgVec3 p_n, cgVec3 p_pt, bool &isParallel)
{
    float t = dot(l_dir, p_n);
    if (std::abs(t) < MIN_DIST)
    {
        isParallel = true;
        return cgVec3(0, 0, 0);
    }
    cgVec3 off = (p_pt - l_orig);
    float d = dot(off, p_n) / t;
    return l_dir * d + l_orig;
}
} // namespace

void geo::aabb::calcAllCorners()
{
    allCorners[0] = cgVec3(c1.x, c1.y, c1.z);
    allCorners[1] = cgVec3(c2.x, c1.y, c1.z);
    allCorners[2] = cgVec3(c1.x, c2.y, c1.z);
    allCorners[3] = cgVec3(c2.x, c2.y, c1.z);
    allCorners[4] = cgVec3(c1.x, c1.y, c2.z);
    allCorners[5] = cgVec3(c2.x, c1.y, c2.z);
    allCorners[6] = cgVec3(c1.x, c2.y, c2.z);
    allCorners[7] = cgVec3(c2.x, c2.y, c2.z);
}

void geo::aabb::calcTransformedAABB(cgMat4 &transform, cgVec3 &c1_t, cgVec3 &c2_t)
{
    cgVec4 init = cgVec4(allCorners[0]);
    cgVec4 initTrans = transform * init;
    c1_t = cgVec3(initTrans.x, initTrans.y, initTrans.z);
    c2_t = cgVec3(initTrans.x, initTrans.y, initTrans.z);

    for (int i = 0; i < 8; i++)
    {
        cgVec4 temp = cgVec4(allCorners[i]);
        cgVec4 tempTrans = transform * temp;

        cgVec3 curr = cgVec3(tempTrans.x, tempTrans.y, tempTrans.z);
        c1_t.x = (curr.x < c1_t.x) ? curr.x : c1_t.x;
        c1_t.y = (curr.y < c1_t.y) ? curr.y : c1_t.y;
        c1_t.z = (curr.z < c1_t.z) ? curr.z : c1_t.z;

        c2_t.x = (curr.x > c2_t.x) ? curr.x : c2_t.x;
        c2_t.y = (curr.y > c2_t.y) ? curr.y : c2_t.y;
        c2_t.z = (curr.z > c2_t.z) ? curr.z : c2_t.z;
    }
}

float geo::aabb::calcSurfaceArea()
{
    float dx = c2.x - c1.x;
    float dy = c2.y - c1.y;
    float dz = c2.z - c1.z;
    return 2 * (dx * dy + dx * dz + dy * dz);
}

float geo::aabb::calcVolume()
{
    float dx = c2.x - c1.x;
    float dy = c2.y - c1.y;
    float dz = c2.z - c1.z;
    return dx * dy * dz;
}

geo::aabb geo::getBoundingBoxOfPtCloud(std::vector<cgVec3> &points)
{
    if (points.size() < 2)
    {
        return geo::aabb();
    }
    geo::aabb bbox;

    cgVec3 init = points[0];
    bbox.c1 = init;
    bbox.c2 = init;

    for (cgVec3 curr : points)
    {
        bbox.c1.x = (curr.x < bbox.c1.x) * curr.x + (!(curr.x < bbox.c1.x)) * bbox.c1.x;
        bbox.c1.y = (curr.y < bbox.c1.y) * curr.y + (!(curr.y < bbox.c1.y)) * bbox.c1.y;
        bbox.c1.z = (curr.z < bbox.c1.z) * curr.z + (!(curr.z < bbox.c1.z)) * bbox.c1.z;

        bbox.c2.x = (curr.x > bbox.c2.x) * curr.x + (!(curr.x > bbox.c2.x)) * bbox.c2.x;
        bbox.c2.y = (curr.y > bbox.c2.y) * curr.y + (!(curr.y > bbox.c2.y)) * bbox.c2.y;
        bbox.c2.z = (curr.z > bbox.c2.z) * curr.z + (!(curr.z > bbox.c2.z)) * bbox.c2.z;
    }
    bbox.calcAllCorners();
    return bbox;
}

geo::aabb geo::getBoundingBoxOfPtCloud(cgVec3 *points, int numPoints)
{
    if (numPoints < 2)
    {
        return geo::aabb();
    }
    geo::aabb bbox;

    cgVec3 init = points[0];
    bbox.c1 = init;
    bbox.c2 = init;

    for (int i = 0; i < numPoints; i++)
    {
        cgVec3 curr = points[i];
        bbox.c1.x = (curr.x < bbox.c1.x) * curr.x + (!(curr.x < bbox.c1.x)) * bbox.c1.x;
        bbox.c1.y = (curr.y < bbox.c1.y) * curr.y + (!(curr.y < bbox.c1.y)) * bbox.c1.y;
        bbox.c1.z = (curr.z < bbox.c1.z) * curr.z + (!(curr.z < bbox.c1.z)) * bbox.c1.z;

        bbox.c2.x = (curr.x > bbox.c2.x) * curr.x + (!(curr.x > bbox.c2.x)) * bbox.c2.x;
        bbox.c2.y = (curr.y > bbox.c2.y) * curr.y + (!(curr.y > bbox.c2.y)) * bbox.c2.y;
        bbox.c2.z = (curr.z > bbox.c2.z) * curr.z + (!(curr.z > bbox.c2.z)) * bbox.c2.z;
    }
    bbox.calcAllCorners();
    return bbox;
}

geo::aabb geo::mergeAABBs(const std::vector<geo::aabb> &aabbs)
{
    geo::aabb merged = aabbs[0];
    for (const aabb &bbox : aabbs)
    {
        merged.c1 = {
            std::min<float>(merged.c1.x, bbox.c1.x),
            std::min<float>(merged.c1.y, bbox.c1.y),
            std::min<float>(merged.c1.z, bbox.c1.z)};
        merged.c2 = {
            std::max<float>(merged.c2.x, bbox.c2.x),
            std::max<float>(merged.c2.y, bbox.c2.y),
            std::max<float>(merged.c2.z, bbox.c2.z)};
    }
    return merged;
}

geo::aabb geo::mergeAABBs(geo::aabb *aabbs, int numAABBs)
{
    geo::aabb merged = aabbs[0];
    if (numAABBs > 1)
    {
        for (int i = 1; i < numAABBs; i++)
        {
            merged.c1 = {
                std::min<float>(merged.c1.x, aabbs[i].c1.x),
                std::min<float>(merged.c1.y, aabbs[i].c1.y),
                std::min<float>(merged.c1.z, aabbs[i].c1.z)};
            merged.c2 = {
                std::max<float>(merged.c2.x, aabbs[i].c2.x),
                std::max<float>(merged.c2.y, aabbs[i].c2.y),
                std::max<float>(merged.c2.z, aabbs[i].c2.z)};
        }
    }
    return merged;
}

void geo::getOnionSkinAABBs(geo::aabb aabb_outer, geo::aabb aabb_inner, geo::aabb *target)
{
    target[0] = geo::aabb(cgVec3(aabb_outer.c1.x, aabb_outer.c1.y, aabb_outer.c1.z), cgVec3(aabb_outer.c2.x, aabb_outer.c2.y, aabb_inner.c1.z));
    target[1] = geo::aabb(cgVec3(aabb_outer.c1.x, aabb_outer.c1.y, aabb_inner.c2.z), cgVec3(aabb_outer.c2.x, aabb_outer.c2.y, aabb_outer.c2.z));
    target[2] = geo::aabb(cgVec3(aabb_outer.c1.x, aabb_outer.c1.y, aabb_inner.c1.z), cgVec3(aabb_inner.c1.x, aabb_outer.c2.y, aabb_inner.c2.z));
    target[3] = geo::aabb(cgVec3(aabb_inner.c2.x, aabb_outer.c1.y, aabb_inner.c1.z), cgVec3(aabb_outer.c2.x, aabb_outer.c2.y, aabb_inner.c2.z));
    target[4] = geo::aabb(cgVec3(aabb_inner.c1.x, aabb_outer.c1.y, aabb_inner.c1.z), cgVec3(aabb_inner.c2.x, aabb_inner.c1.y, aabb_inner.c2.z));
    target[5] = geo::aabb(cgVec3(aabb_inner.c1.x, aabb_inner.c2.y, aabb_inner.c1.z), cgVec3(aabb_inner.c2.x, aabb_outer.c2.y, aabb_inner.c2.z));
}

void geo::aabb::print()
{
    std::cout << "c1: ";
    c1.print();
    std::cout << "c2: ";
    c2.print();
}

bool geo::areColliding(geo::aabb &aabb1, geo::aabb &aabb2)
{
    return ((((aabb1.c1.x >= aabb2.c1.x) && (aabb1.c1.x <= aabb2.c2.x)) ||
             ((aabb1.c2.x >= aabb2.c1.x) && (aabb1.c2.x <= aabb2.c2.x))) ||
            (((aabb2.c1.x >= aabb1.c1.x) && (aabb2.c1.x <= aabb1.c2.x)) ||
             ((aabb2.c2.x >= aabb1.c1.x) && (aabb2.c2.x <= aabb1.c2.x)))) &&
           ((((aabb1.c1.y >= aabb2.c1.y) && (aabb1.c1.y <= aabb2.c2.y)) ||
             ((aabb1.c2.y >= aabb2.c1.y) && (aabb1.c2.y <= aabb2.c2.y))) ||
            (((aabb2.c1.y >= aabb1.c1.y) && (aabb2.c1.y <= aabb1.c2.y)) ||
             ((aabb2.c2.y >= aabb1.c1.y) && (aabb2.c2.y <= aabb1.c2.y)))) &&
           ((((aabb1.c1.z >= aabb2.c1.z) && (aabb1.c1.z <= aabb2.c2.z)) ||
             ((aabb1.c2.z >= aabb2.c1.z) && (aabb1.c2.z <= aabb2.c2.z))) ||
            (((aabb2.c1.z >= aabb1.c1.z) && (aabb2.c1.z <= aabb1.c2.z)) ||
             ((aabb2.c2.z >= aabb1.c1.z) && (aabb2.c2.z <= aabb1.c2.z))));
}

bool geo::checkLineIntersect(geo::aabb &aabb1, cgVec3 l_orig, cgVec3 l_dir)
{
    cgVec3 p_n;
    cgVec3 p_pt;
    cgVec3 intersect;

    p_n = cgVec3(1, 0, 0);
    p_pt = aabb1.c2;
    bool isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.y >= aabb1.c1.y && intersect.y <= aabb1.c2.y &&
            intersect.z >= aabb1.c1.z && intersect.z <= aabb1.c2.z)
        {
            return true;
        }
    }

    p_n = cgVec3(0, 1, 0);
    isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.x >= aabb1.c1.x && intersect.x <= aabb1.c2.x &&
            intersect.z >= aabb1.c1.z && intersect.z <= aabb1.c2.z)
        {
            return true;
        }
    }

    p_n = cgVec3(0, 0, 1);
    isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.x >= aabb1.c1.x && intersect.x <= aabb1.c2.x &&
            intersect.y >= aabb1.c1.y && intersect.y <= aabb1.c2.y)
        {
            return true;
        }
    }

    p_n = cgVec3(-1, 0, 0);
    p_pt = aabb1.c1;
    isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.y >= aabb1.c1.y && intersect.y <= aabb1.c2.y &&
            intersect.z >= aabb1.c1.z && intersect.z <= aabb1.c2.z)
        {
            return true;
        }
    }

    p_n = cgVec3(0, -1, 0);
    isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.x >= aabb1.c1.x && intersect.x <= aabb1.c2.x &&
            intersect.z >= aabb1.c1.z && intersect.z <= aabb1.c2.z)
        {
            return true;
        }
    }

    p_n = cgVec3(0, 0, -1);
    isParallel = false;
    intersect = linePlaneIntersectionPt(l_orig, l_dir, p_n, p_pt, isParallel);
    if (!isParallel)
    {
        if (intersect.x >= aabb1.c1.x && intersect.x <= aabb1.c2.x &&
            intersect.y >= aabb1.c1.y && intersect.y <= aabb1.c2.y)
        {
            return true;
        }
    }

    return false;
}
