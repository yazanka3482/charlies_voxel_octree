#ifndef GEO_PRIMITIVES_HPP
#define GEO_PRIMITIVES_HPP

#include "AABB.hpp"
#include "cgVec.hpp"

namespace geo {

struct Facet {
    int inds[3] = {0, 0, 0};

    Facet() = default;
    Facet(int i1, int i2, int i3) { inds[0] = i1; inds[1] = i2; inds[2] = i3; }
    Facet(const Facet& f) { inds[0] = f.inds[0]; inds[1] = f.inds[1]; inds[2] = f.inds[2]; }
};

/** Parity / inside-outside: only forward hits with t > tMin. */
bool rayIntersectsTriangleParity(cgVec3& rayOrigin,
                                 cgVec3& rayVector,
                                 cgVec3& tri0,
                                 cgVec3& tri1,
                                 cgVec3& tri2,
                                 float tMin,
                                 cgVec3& outIntersectionPoint);

} // namespace geo

#endif
