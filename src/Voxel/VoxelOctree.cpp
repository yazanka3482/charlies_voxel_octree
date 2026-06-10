#include "VoxelOctree.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stack>

VoxelOctree::VoxelOctree(geo::aabb boundingBox, int numLevels)
{
    this->boundingBox = boundingBox;
    this->octree = BitOctree(numLevels);
    this->octree.m_RootNode.m_Data = 0;
    this->octree.m_RootNode.m_Children = 0;
    this->numLevels = numLevels;
    int64_t voxelRes = 0x1 << numLevels;
    cgVec3 bboxSize = boundingBox.c2 - boundingBox.c1;
    this->maxRes = bboxSize / float(voxelRes);
    numOctrees = 1;
}

VoxelOctree::~VoxelOctree(){
}


uint8_t VoxelOctree::getDataAt(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc)
{
    std::shared_lock<std::shared_mutex> lock(octreeMutex);
    return this->octree.getDataAt(indX, indY, indZ, maxlevel, cellDesc);
}

uint8_t VoxelOctree::getDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc)
{
    return this->octree.getDataAt(indX, indY, indZ, maxlevel, cellDesc);
}

uint8_t VoxelOctree::getDataAtClamped(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc)
{
    int64_t maxInd = this->getMaxIndex();
    int64_t indXAdj = std::max(indX, int64_t(0));
    int64_t indYAdj = std::max(indY, int64_t(0));
    int64_t indZAdj = std::max(indZ, int64_t(0));
    indXAdj = std::min(indXAdj, maxInd);
    indYAdj = std::min(indYAdj, maxInd);
    indZAdj = std::min(indZAdj, maxInd);
    std::shared_lock<std::shared_mutex> lock(octreeMutex);
    return this->octree.getDataAt(indXAdj, indYAdj, indZAdj, maxlevel, cellDesc);
}

uint8_t VoxelOctree::getDataAtClamped_unlocked(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc)
{
    int64_t maxInd = this->getMaxIndex();
    int64_t indXAdj = std::max(indX, int64_t(0));
    int64_t indYAdj = std::max(indY, int64_t(0));
    int64_t indZAdj = std::max(indZ, int64_t(0));
    indXAdj = std::min(indXAdj, maxInd);
    indYAdj = std::min(indYAdj, maxInd);
    indZAdj = std::min(indZAdj, maxInd);
    return this->octree.getDataAt(indXAdj, indYAdj, indZAdj, maxlevel, cellDesc);
}

void VoxelOctree::setDataAt(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data)
{
    std::unique_lock<std::shared_mutex> lock(octreeMutex);
    this->octree.setDataAt(indX, indY, indZ, level, data);
}

void VoxelOctree::setDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data)
{
    this->octree.setDataAt(indX, indY, indZ, level, data);
}

void VoxelOctree::replaceDataAt_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data, uint8_t dataToIgnore){
    this->octree.replaceDataAt(indX, indY, indZ, level, data, dataToIgnore);
}

cgVec3 VoxelOctree::getPositionOfIndex(int64_t indX, int64_t indY, int64_t indZ, int level)
{
    cgVec3 pos;
    pos.x = this->boundingBox.c1.x + this->maxRes.x * float(indX);
    pos.y = this->boundingBox.c1.y + this->maxRes.y * float(indY);
    pos.z = this->boundingBox.c1.z + this->maxRes.z * float(indZ);
    return pos;
}

void VoxelOctree::getIndexOfPosition(cgVec3 position, int64_t &indX, int64_t &indY, int64_t &indZ)
{
    indX = int64_t((position.x - boundingBox.c1.x) / this->maxRes.x);
    indY = int64_t((position.y - boundingBox.c1.y) / this->maxRes.y);
    indZ = int64_t((position.z - boundingBox.c1.z) / this->maxRes.z);
}

void VoxelOctree::getIndexOfPositionBounded(cgVec3 position, int64_t &indX, int64_t &indY, int64_t &indZ)
{
    int64_t tempX = int64_t((position.x - boundingBox.c1.x) / this->maxRes.x);
    int64_t tempY = int64_t((position.y - boundingBox.c1.y) / this->maxRes.y);
    int64_t tempZ = int64_t((position.z - boundingBox.c1.z) / this->maxRes.z);

    int64_t mi = this->getMaxIndex() - 1;
    if (tempX >= mi)
        tempX = mi;
    if (tempY >= mi)
        tempY = mi;
    if (tempZ >= mi)
        tempZ = mi;

    if (tempX <= 0)
        tempX = 0;
    if (tempY <= 0)
        tempY = 0;
    if (tempZ <= 0)
        tempZ = 0;

    indX = int64_t(tempX);
    indY = int64_t(tempY);
    indZ = int64_t(tempZ);
}

void VoxelOctree::getCornerPositions(int64_t indX, int64_t indY, int64_t indZ, int level, cgVec3 &c0, cgVec3 &c1)
{
    c0 = getPositionOfIndex(indX, indY, indZ, level);
    c1 = c0 + getVoxelSize(level);
}

cgVec3 VoxelOctree::getVoxelSize(int level)
{
    return maxRes * float(int(0x1 << level));
}

bool VoxelOctree::pointIsFilled(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t &type)
{
    OctreeCellDescriptor cellDesc;
    int temp = level - 1;
    int sl = (0x1) << temp;
    std::shared_lock<std::shared_mutex> lock(octreeMutex);
    type = this->getDataAt_unlocked(indX, indY, indZ, temp, cellDesc);
    return type > 0;
}

bool VoxelOctree::pointIsFilled_unlocked(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t &type)
{
    OctreeCellDescriptor cellDesc;
    int temp = level - 1;
    int sl = (0x1) << temp;
    type = this->getDataAt_unlocked(indX, indY, indZ, temp, cellDesc);
    return type > 0;
}

void VoxelOctree::setRangeTo(cgVec3 btmleft, cgVec3 topright, int type)
{
    int64_t spx, spy, spz;
    getIndexOfPositionBounded({btmleft.x, btmleft.z, btmleft.y}, spx, spy, spz);
    int64_t epx, epy, epz;
    getIndexOfPositionBounded({topright.x, topright.z, topright.y}, epx, epy, epz);
    // this->octreeMutex.lock();
    std::unique_lock<std::shared_mutex> lock(octreeMutex);
    for (int64_t i = spx; i < epx; i++)
    {
        for (int64_t j = spy; j < epy; j++)
        {
            for (int64_t k = spz; k < epz; k++)
            {
                this->octree.setDataAt(i, j, k, 0, type);
            }
        }
    }
}

void VoxelOctree::setSphereTo(cgVec3 center, float radius, int type)
{

    cgVec3 btmleft = center - cgVec3(radius, radius, radius);
    cgVec3 topright = center + cgVec3(radius, radius, radius);
    int64_t spx, spy, spz;
    getIndexOfPositionBounded({btmleft.x, btmleft.z, btmleft.y}, spx, spy, spz);
    int64_t epx, epy, epz;
    getIndexOfPositionBounded({topright.x, topright.z, topright.y}, epx, epy, epz);
    // this->octreeMutex.lock();
    std::unique_lock<std::shared_mutex> lock(octreeMutex);
    for (int64_t i = spx; i < epx; i++)
    {
        for (int64_t j = spy; j < epy; j++)
        {
            for (int64_t k = spz; k < epz; k++)
            {
                cgVec3 worldPos = getPositionOfIndex(i, j, k, 0);
                if ((worldPos - cgVec3(center.x, center.z, center.y)).norm() <= radius)
                {
                    this->octree.setDataAt(i, j, k, 0, type);
                }
            }
        }
    }
}

void VoxelOctree::setSphereTo(cgVec3 center, float radius, int type, int octreeLevel)
{
    int octreeLevelAdj = std::min(octreeLevel, this->getMaxLevels());
    int sl = (0x1 << octreeLevelAdj);
    cgVec3 btmleft = center - cgVec3(radius, radius, radius);
    cgVec3 topright = center + cgVec3(radius, radius, radius);
    int64_t spx, spy, spz;
    getIndexOfPositionBounded({btmleft.x, btmleft.z, btmleft.y}, spx, spy, spz);
    int64_t epx, epy, epz;
    getIndexOfPositionBounded({topright.x, topright.z, topright.y}, epx, epy, epz);
    std::unique_lock<std::shared_mutex> lock(octreeMutex);

    for (int64_t i = spx; i < epx; i += sl)
    {
        for (int64_t j = spy; j < epy; j += sl)
        {
            for (int64_t k = spz; k < epz; k += sl)
            {
                cgVec3 worldPos = getPositionOfIndex(i, j, k, 0);
                if ((worldPos - cgVec3(center.x, center.z, center.y)).norm() <= radius)
                {
                    this->octree.setDataAt(i, j, k, octreeLevelAdj, type);
                }
            }
        }
    }
}

void VoxelOctree::replaceSphere(cgVec3 center, float radius, int type, int typeToIgnore, int octreeLevel)
{
    int octreeLevelAdj = std::min(octreeLevel, this->getMaxLevels());
    int sl = (0x1 << octreeLevelAdj);
    cgVec3 btmleft = center - cgVec3(radius, radius, radius);
    cgVec3 topright = center + cgVec3(radius, radius, radius);
    int64_t spx, spy, spz;
    getIndexOfPositionBounded({btmleft.x, btmleft.z, btmleft.y}, spx, spy, spz);
    int64_t epx, epy, epz;
    getIndexOfPositionBounded({topright.x, topright.z, topright.y}, epx, epy, epz);
    std::unique_lock<std::shared_mutex> lock(octreeMutex);

    for (int64_t i = spx; i < epx; i += sl)
    {
        for (int64_t j = spy; j < epy; j += sl)
        {
            for (int64_t k = spz; k < epz; k += sl)
            {
                cgVec3 worldPos = getPositionOfIndex(i, j, k, 0);
                if ((worldPos - cgVec3(center.x, center.z, center.y)).norm() <= radius)
                {
                    replaceDataAt_unlocked(i, j, k, octreeLevelAdj, type, typeToIgnore);
                }
            }
        }
    }
}

void VoxelOctree::saveVerticalSliceToJPG(std::string filename, int xValue)
{
    (void)filename;
    (void)xValue;
}

uint64_t VoxelOctree::getMemoryUsage(){
    uint64_t memoryUsage = 0;
    memoryUsage += octree.m_BranchNodes.n_AllocatedChunks * sizeof(BitOctree::EightBranchNodes);
    memoryUsage += octree.m_LeafNodes.n_AllocatedChunks * sizeof(BitOctree::EightLeafNodes);
    return memoryUsage;
}

void VoxelOctree::saveToFile(std::string filename){
    std::ofstream file;
    std::string fname_adj = filename;
    if (fname_adj.size() < 6 || fname_adj.substr(fname_adj.size() - 6) != ".voxel")
        fname_adj += ".voxel";
    file.open(fname_adj, std::ios::binary);
    std::shared_lock<std::shared_mutex> lock(octreeMutex);
    if(!file.is_open()){
        std::cout << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    this->octree.writeToFile(file);
    file.close();
}

void VoxelOctree::loadFromFile(std::ifstream &file){
    std::cout << "Loading VoxelOctree from file\n";
    std::unique_lock<std::shared_mutex> lock(octreeMutex);
    this->octree.readFromFile(file);
    this->isLoaded = true;
}

void VoxelOctree::loadFromFile(std::string filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        std::cout << "Could not open file: " << filename << std::endl;
        return;
    }
    loadFromFile(file);
    file.close();
}

void VoxelOctree::setRotatedAABBTo(cgVec3 position, quat rotation, geo::aabb bbox, cgVec3 rotCenter, int octreeLevel, int type)
{
    int octreeLevelAdj = std::min(octreeLevel, this->getMaxLevels());
    int sl = (0x1 << octreeLevelAdj);

    // Calculate bounds of the AABB in world space
    geo::aabb bbox_rotated;
    bbox.calcAllCorners();
    cgMat4 aabbTransform = trans(-rotCenter.x,-rotCenter.y,-rotCenter.z)*rotation.getRotMatrix()*trans(-rotCenter.x,-rotCenter.y,-rotCenter.z);
    bbox.calcTransformedAABB(aabbTransform, bbox_rotated.c1, bbox_rotated.c2);

    cgVec3 btmleft = bbox_rotated.c1 + position;
    cgVec3 topright = bbox_rotated.c2 + position;

    // Get voxel indices for bounds
    int64_t spx, spy, spz;
    getIndexOfPositionBounded({btmleft.x, btmleft.y, btmleft.z}, spx, spy, spz);
    int64_t epx, epy, epz;
    getIndexOfPositionBounded({topright.x, topright.y, topright.z}, epx, epy, epz);
    std::unique_lock<std::shared_mutex> lock(octreeMutex);
    quat inverse_rotation = rotation.inv();

    // Iterate through voxels in bounds
    for (int64_t i = spx; i <= epx; i += sl)
    {
        for (int64_t j = spy; j <= epy; j += sl)
        {
            for (int64_t k = spz; k <= epz; k += sl)
            {
                cgVec3 worldPos = getPositionOfIndex(i, j, k, 0);
                
                // Transform point to local space and rotate
                cgVec3 localPos = worldPos - position;
                localPos = localPos - rotCenter;
                localPos = inverse_rotation.rotate(localPos);
                localPos = localPos + rotCenter;

                // Check if point is inside AABB
                if(localPos.x >= bbox.c1.x && localPos.x <= bbox.c2.x &&
                   localPos.y >= bbox.c1.y && localPos.y <= bbox.c2.y &&
                   localPos.z >= bbox.c1.z && localPos.z <= bbox.c2.z) {
                    this->octree.setDataAt(i, k, j, octreeLevelAdj, type);
                }
            }
        }
    }
}