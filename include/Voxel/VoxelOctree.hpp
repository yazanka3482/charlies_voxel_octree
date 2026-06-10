#ifndef VOXELOCTREE
#define VOXELOCTREE
#include "Octree.hpp"
#include "AABB.hpp"
#include "cgVec.hpp"
#include <mutex>
#include <shared_mutex>
#include <stdint.h>

/**
 * @brief Sparse bit octree for voxel data over a bounded 3D domain.
 *
 * Voxel indexing:
 *   deepest voxel in octree = index 0 (smallest voxel)
 *   levels are indexed from 0 to numLevels-1
 *   voxel indices use unsigned integers where the rightmost bit is level 0
 */
class VoxelOctree
{

public:
    VoxelOctree(geo::aabb boundingBox, int numLevels);

    ~VoxelOctree();

    uint8_t getDataAt(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc);

    uint8_t getDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc);

    uint8_t getDataAtClamped(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc);

    uint8_t getDataAtClamped_unlocked(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc);

    void setDataAt(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data);

    void setDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data);

    void replaceDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data, uint8_t dataToIgnore);

    cgVec3 getPositionOfIndex(int64_t indX, int64_t indY, int64_t indZ, int level);

    void getIndexOfPosition(cgVec3 position, int64_t &indX, int64_t &indY, int64_t &indZ);

    void getIndexOfPositionBounded(cgVec3 position, int64_t &indX, int64_t &indY, int64_t &indZ);

    void getCornerPositions(int64_t indX, int64_t indY, int64_t indZ, int level, cgVec3 &c0, cgVec3 &c1);

    cgVec3 getVoxelSize(int level);

    mutable std::shared_mutex octreeMutex;

    int getMaxLevels()
    {
        return numLevels;
    }

    int64_t getMaxIndex()
    {
        return int64_t(0x1 << (numLevels));
    }

    int64_t getLevelOffset(int level)
    {
        return 0x1 << level;
    }

    int numOctrees = 0;

    bool pointIsFilled(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t &type);

    bool pointIsFilled_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t &type);

    void setRangeTo(cgVec3 btmleft, cgVec3 topright, int type);

    void setSphereTo(cgVec3 center, float radius, int type);

    void setSphereTo(cgVec3 center, float radius, int type, int octreeLevel);

    void setRotatedAABBTo(cgVec3 position, quat rotation, geo::aabb bbox, cgVec3 rotCenter, int octreeLevel, int type);

    void replaceSphere(cgVec3 center, float radius, int type, int typeToIgnore, int octreeLevel);

    void saveVerticalSliceToJPG(std::string filename, int xValue);

    uint64_t getMemoryUsage();

    void saveToFile(std::string filename);

    void loadFromFile(std::ifstream &file);

    void loadFromFile(std::string filename);

    geo::aabb boundingBox;

    cgVec3 maxRes;

    std::atomic_bool isLoaded = false;
    
    BitOctree octree = BitOctree();

private:

    int numLevels;

};

#endif /* VOXELOCTREE */
