#include "OctreeGeneration.hpp"
#include "CumulativeProfiler.hpp"
#include "Stopwatch.hpp"
#include <stdint.h>

void OctreeGenerater::getVolumeDescBatch(const OctreeVoxelQuery* queries, OctreeVoxelGenDesc* out, size_t count)
{
    for (size_t i = 0; i < count; i++)
        out[i] = getVolumeDesc(queries[i].volume);
}

static void applyBatchResults(
    VoxelOctree* vworld,
    int level,
    const OctreeVoxelQuery* queries,
    const OctreeVoxelGenDesc* results,
    size_t count,
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& edgeVoxels,
    Stopwatch& sw)
{
    static const int kSetData = CumulativeProfiler::intern("octree/setDataAt");
    for (size_t i = 0; i < count; i++)
    {
        const OctreeVoxelGenDesc& volDesc = results[i];
        if (volDesc.loc == intersecting)
            edgeVoxels.push_back(std::make_tuple(queries[i].indX, queries[i].indY, queries[i].indZ));
        if (!volDesc.noneType)
        {
            sw.start();
            vworld->setDataAt_unlocked(queries[i].indX, queries[i].indY, queries[i].indZ, level, volDesc.type);
            CumulativeProfiler::addSample(kSetData, sw.stop_time());
        }
    }
}

void OctreeGenerater::addToWorld(VoxelOctree* vworld, int voxelizationLevel){
    PROFILE_SCOPE("voxel/octree_addToWorld");
    static const int kGetVolumeDesc = CumulativeProfiler::intern("octree/getVolumeDesc_calls");
    static const int kCornerPos = CumulativeProfiler::intern("octree/getCornerPositions");
    Stopwatch sw;
    m_MinVoxelizationLevel = voxelizationLevel;

    std::vector<std::vector<std::tuple<uint64_t,uint64_t,uint64_t>>> edgeVoxels = std::vector<std::vector<std::tuple<uint64_t,uint64_t,uint64_t>>>(vworld->getMaxLevels()+1);

    std::unique_lock<std::shared_mutex> lock(vworld->octreeMutex);
    int currLevel = vworld->getMaxLevels()-1;

    std::vector<OctreeVoxelQuery> batchQueries;
    std::vector<OctreeVoxelGenDesc> batchResults;
    batchQueries.reserve(1024);
    batchResults.reserve(1024);

    auto flushBatch = [&](int level) {
        if (batchQueries.empty())
            return;
        m_CurrOctreeLevel = level;
        sw.start();
        getVolumeDescBatch(batchQueries.data(), batchResults.data(), batchQueries.size());
        CumulativeProfiler::addSample(kGetVolumeDesc, sw.stop_time());
        applyBatchResults(vworld, level, batchQueries.data(), batchResults.data(), batchQueries.size(), edgeVoxels[level], sw);
        batchQueries.clear();
    };

    for(int ix = 0; ix < 2; ix++){
        for(int iy = 0; iy < 2; iy++){
            for(int iz = 0; iz < 2; iz++){
                uint64_t indx = ix<<currLevel;
                uint64_t indy = iy<<currLevel;
                uint64_t indz = iz<<currLevel;

                OctreeVoxelQuery query;
                query.indX = indx;
                query.indY = indy;
                query.indZ = indz;
                sw.start();
                vworld->getCornerPositions(indx,indy,indz,currLevel,query.volume.c1,query.volume.c2);
                CumulativeProfiler::addSample(kCornerPos, sw.stop_time());
                batchQueries.push_back(query);
            }
        }
    }
    batchResults.resize(batchQueries.size());
    flushBatch(currLevel);

    currLevel--;
    while(currLevel>=(voxelizationLevel)){
        std::cout << "Edge voxels at level "<<currLevel+1<<": "<< edgeVoxels[currLevel+1].size()<<"\n";
        batchQueries.clear();
        for(auto edgeVoxel : edgeVoxels[currLevel+1]){
            uint64_t u_voxX = std::get<0>(edgeVoxel);
            uint64_t u_voxY = std::get<1>(edgeVoxel);
            uint64_t u_voxZ = std::get<2>(edgeVoxel);
            for(int ix = 0; ix < 2; ix++){
                for(int iy = 0; iy < 2; iy++){
                    for(int iz = 0; iz < 2; iz++){
                        uint64_t indx = uint64_t(ix<<currLevel) + uint64_t(u_voxX);
                        uint64_t indy = uint64_t(iy<<currLevel) + uint64_t(u_voxY);
                        uint64_t indz = uint64_t(iz<<currLevel) + uint64_t(u_voxZ);

                        OctreeVoxelQuery query;
                        query.indX = indx;
                        query.indY = indy;
                        query.indZ = indz;
                        sw.start();
                        vworld->getCornerPositions(indx,indy,indz,currLevel,query.volume.c1,query.volume.c2);
                        CumulativeProfiler::addSample(kCornerPos, sw.stop_time());
                        batchQueries.push_back(query);
                    }
                }
            }
        }
        batchResults.resize(batchQueries.size());
        flushBatch(currLevel);
        edgeVoxels[currLevel+1].clear();
        currLevel--;
    }
    std::cout << "Octree generation completed" << std::endl;
}
