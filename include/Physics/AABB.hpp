#ifndef AABB_HPP
#define AABB_HPP

#include <algorithm>
#include <cstring>
#include <vector>
#include "cgVec.hpp"

namespace geo
{

/**
 * @brief Axis aligned bounding box
 */
struct aabb
{
    cgVec3 c1; // two corners
    cgVec3 c2; // this corner always has the larger x,y,z components

    aabb() {}

    aabb(const aabb &bb2)
    {
        memcpy(this, &bb2, sizeof(aabb));
    }

    aabb(cgVec3 corner1, cgVec3 corner2)
    {
        this->c1 = corner1;
        this->c2 = corner2;
    }

    void print();
    void calcAllCorners();

    float calcSurfaceArea();

    float calcVolume();

    /**
     * @brief Calculates a rotated version of this
     *
     * @param transform the matrix transform to be applied to aabb vertices
     * @param c1_t target corner 1 - for your transformed aabb
     * @param c2_t target corner 2 - for your transformed aabb
     */
    void calcTransformedAABB(cgMat4 &transform, cgVec3 &c1_t, cgVec3 &c2_t);

    cgVec3 getCenter()
    {
        return (c1 + c2) * 0.5;
    }

    cgVec3 allCorners[8]; // used for fast rotations
};

aabb getBoundingBoxOfPtCloud(std::vector<cgVec3> &points);

aabb getBoundingBoxOfPtCloud(cgVec3 *points, int numPoints);

inline aabb mergeAABBs(aabb &aabb1, aabb &aabb2)
{
    aabb out;
    out.c1 = {
        std::min<float>(aabb1.c1.x, aabb2.c1.x),
        std::min<float>(aabb1.c1.y, aabb2.c1.y),
        std::min<float>(aabb1.c1.z, aabb2.c1.z)};
    out.c2 = {
        std::max<float>(aabb1.c2.x, aabb2.c2.x),
        std::max<float>(aabb1.c2.y, aabb2.c2.y),
        std::max<float>(aabb1.c2.z, aabb2.c2.z)};
    return out;
}

aabb mergeAABBs(const std::vector<aabb> &aabbs);

aabb mergeAABBs(aabb *aabbs, int numAABBs);

void getOnionSkinAABBs(aabb aabb_outer, aabb aabb_inner, aabb *target);

bool areColliding(aabb &aabb1, aabb &aabb2);

bool checkLineIntersect(aabb &aabb1, cgVec3 l_orig, cgVec3 l_dir);

} // namespace geo

#endif /* AABB_HPP */
