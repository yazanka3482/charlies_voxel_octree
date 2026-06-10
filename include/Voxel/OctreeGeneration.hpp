#ifndef OCTREEGENERATION
#define OCTREEGENERATION

/**
 * @file OctreeGeneration.hpp
 * @author charlie callahan
 * @brief algorithm for generating octrees based on objects that can indicate if a point is inside of its volume
 * @version 0.1
 * @date 2023-10-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "VoxelOctree.hpp"
#include "AABB.hpp"

/**
 * @brief For generating an octree from a volume, this describes where a cube in that volume lies
 * 
 */
enum OctreeVolumeLocation{
    inside,
    outside,
    intersecting
};

/**
 * @brief no good name for this, just what should get returned by "getVolumeLocation"
 * 
 */
struct OctreeVoxelGenDesc{
    OctreeVolumeLocation loc;
    uint8_t type;
    bool noneType=false; //set to true to leave the octree unchanged at this voxel
};

struct OctreeVoxelQuery{
    uint64_t indX = 0;
    uint64_t indY = 0;
    uint64_t indZ = 0;
    geo::aabb volume;
};

/**
 * @brief This represents an object that can be added to the voxel octree. something like a height map, mesh or something like that
 * 
 */
class OctreeGenerater {
    public:
    virtual ~OctreeGenerater(){};

    /**
     * @brief Adds the generater to the world
     * 
     * @param vworld 
     * @param voxelizationLevel the highest octree resolution to mesh at 0 = max resolution
     */
    void addToWorld(VoxelOctree* vworld, int voxelizationLevel);
    
protected:
    /** Set by addToWorld before each getVolumeDesc call. */
    int m_CurrOctreeLevel = 0;
    int m_MinVoxelizationLevel = 0;
    
    /**
     * @brief Get the descriptor for the volume given by voxel
     * ie: is it intersecting, fully inside or fully outside the surface
     * and what type should it be given
     * 
     * @param voxel 
     * @return OctreeVoxelGenDesc 
     */
    virtual OctreeVoxelGenDesc getVolumeDesc(geo::aabb voxel)=0;

    /**
     * @brief Classify many voxels at once. Default implementation calls getVolumeDesc per voxel.
     */
    virtual void getVolumeDescBatch(const OctreeVoxelQuery* queries, OctreeVoxelGenDesc* out, size_t count);
};

#endif