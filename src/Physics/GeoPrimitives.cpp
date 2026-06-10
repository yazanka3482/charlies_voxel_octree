#include "GeoPrimitives.hpp"

namespace geo {

bool rayIntersectsTriangleParity(cgVec3& rayOrigin,
                                 cgVec3& rayVector,
                                 cgVec3& tri0,
                                 cgVec3& tri1,
                                 cgVec3& tri2,
                                 float tMin,
                                 cgVec3& outIntersectionPoint)
{
    const float EPSILON = 1e-7f;
    const float BARY_EPS = 1e-5f;
    cgVec3 edge1 = tri1 - tri0;
    cgVec3 edge2 = tri2 - tri0;
    cgVec3 h = cross(rayVector, edge2);
    float a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;
    float f = 1.0f / a;
    cgVec3 s = rayOrigin - tri0;
    float u = f * dot(s, h);
    if (u < -BARY_EPS || u > 1.0f + BARY_EPS)
        return false;
    cgVec3 q = cross(s, edge1);
    float v = f * dot(rayVector, q);
    if (v < -BARY_EPS || u + v > 1.0f + BARY_EPS)
        return false;
    float t = f * dot(edge2, q);
    if (t > tMin)
    {
        outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    return false;
}

} // namespace geo
