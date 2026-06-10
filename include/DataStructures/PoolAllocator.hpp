#ifndef POOLALLOCATOR
#define POOLALLOCATOR
#include <stdint.h>
#include <assert.h>
#include <fstream>

/**
 * @brief This uses 32bit pointers, so it will work if your data is < 8bytes, but at the cost of being unable to store more than 4.2B data instances
 * (it can store more than 4.2B bytes)
 */

template<typename T>
struct PoolAllocator32b {

    union Chunk {
        Chunk(){T();}
        uint32_t m_Next;
        T m_Data;
    };

    Chunk** m_Pools = nullptr;

    int m_NumPools;
    int m_MaxNumPools=0;

    uint32_t n_AllocatedChunks = 0; //just for profiling
    uint32_t m_FreeChunk = 0;
    uint32_t m_LastPoolHighWaterMark = 0; // highest used chunk index in the last pool

    // this bitmask represents the bits of an index into a pool array (the upper bits)
    uint32_t m_InPoolBitmask = 0;

    // this one represents the bits which index into the pools, (the lower bits)
    uint32_t m_PoolIndexBitmask = 0;

    // number of chunks per pool = 2^m_Log2PoolSize
    int m_Log2PoolSize = 0;
    PoolAllocator32b(){
        m_MaxNumPools=0;
    }

    /**
     * @param log2PoolSize size of a single pool
     * @param maxNumberOfPools maximum number of pools, they will be allocated as needed
     */
    PoolAllocator32b(int log2PoolSize, int maxNumberOfPools){
        assert(maxNumberOfPools>0);
        m_Log2PoolSize = log2PoolSize;
        m_InPoolBitmask = 0x0;
        for(int i = 0; i < m_Log2PoolSize; i++){
            m_InPoolBitmask = m_InPoolBitmask | (0x1 << i); // all ones for the first m_Log2PoolSize bits
        }
        m_PoolIndexBitmask = ~m_InPoolBitmask; // all zeros below m_Log2PoolSize bits

        m_Pools = new Chunk*[maxNumberOfPools];
        for (int i = 0; i < maxNumberOfPools; i++){
            m_Pools[i] = nullptr;
        }
        m_MaxNumPools = maxNumberOfPools;

        //setup the first pool
        m_Pools[0] = new Chunk[0x1<<m_Log2PoolSize];
        m_FreeChunk = _initializePool(m_Pools[0], 0);
        m_NumPools = 1;
        n_AllocatedChunks = 0;
    }

    // returns the first index into the pool, only use this on the last pool
    uint32_t _initializePool(Chunk* pool, uint32_t poolIndex){
        // initialize the pool at the end of the pools array
        // purposely skip the first chunk since it will be used as the "null" pointer
        int poolSize = 0x1 << m_Log2PoolSize;
        for (int i = 1; i < poolSize-1; i++){
            uint32_t nextIndex = i+1;
            uint32_t nextIntraPoolIndex = nextIndex | (poolIndex << m_Log2PoolSize);
            pool[i].m_Next = nextIntraPoolIndex;
        }
        pool[poolSize-1].m_Next = (poolIndex << m_Log2PoolSize); //point the last one to "NULL" but keep the pool index
                                                                 //this indicates that this is the last index of the last pool
        return 0x1 | (poolIndex << m_Log2PoolSize);
    }

    void _appendPoolFreeTail(Chunk* pool, uint32_t poolIndex, uint32_t startIdx){
        int poolSize = 0x1 << m_Log2PoolSize;
        for (int i = startIdx; i < poolSize - 1; i++){
            pool[i].m_Next = (i + 1) | (poolIndex << m_Log2PoolSize);
        }
        pool[poolSize - 1].m_Next = (poolIndex << m_Log2PoolSize);

        uint32_t ptr = m_FreeChunk;
        if (((ptr & m_PoolIndexBitmask) >> m_Log2PoolSize) != poolIndex){
            if ((ptr & m_InPoolBitmask) == 0){
                m_FreeChunk = startIdx | (poolIndex << m_Log2PoolSize);
            }
            return;
        }
        while (true){
            uint32_t inPool = ptr & m_InPoolBitmask;
            uint32_t next = pool[inPool].m_Next;
            if ((next & m_InPoolBitmask) == 0){
                pool[inPool].m_Next = startIdx | (poolIndex << m_Log2PoolSize);
                break;
            }
            ptr = next;
        }
    }

    uint32_t _poolHighWaterMark(int poolIndex) const {
        if (poolIndex < m_NumPools - 1){
            return (0x1u << m_Log2PoolSize) - 1;
        }
        return m_LastPoolHighWaterMark;
    }

    /**
     * @brief returns the next available 32b pointer 
     * returns 0 if there are no available chunks
     */
    uint32_t allocate(){
        uint32_t current_free_chunk = m_FreeChunk;
        if( (current_free_chunk & m_InPoolBitmask) == 0 ){ // hit the last chunk in the last pool, need to allocate a new pool
            if(m_NumPools >= m_MaxNumPools) {
                return 0; // pool allocator is full
            }

            m_LastPoolHighWaterMark = (0x1u << m_Log2PoolSize) - 1;
            m_Pools[m_NumPools] = new Chunk[0x1<<m_Log2PoolSize];
            // intializes the new pool and points the free chunk to the first index of the new pool
            current_free_chunk = _initializePool(m_Pools[m_NumPools], m_NumPools);
            m_NumPools++;
            m_LastPoolHighWaterMark = 0;
        }
        uint32_t poolsIndex = (current_free_chunk & m_PoolIndexBitmask) >> m_Log2PoolSize;
        uint32_t inPoolIndex = current_free_chunk & m_InPoolBitmask;
        m_FreeChunk = m_Pools[poolsIndex][inPoolIndex].m_Next;
        if (poolsIndex == (uint32_t)(m_NumPools - 1) && inPoolIndex > m_LastPoolHighWaterMark){
            m_LastPoolHighWaterMark = inPoolIndex;
        }
        n_AllocatedChunks++;
        return current_free_chunk;
    }

    void deallocate(uint32_t chunkPtr){
        // prepend the chunkPtr to the free list
        uint32_t poolsIndex = (chunkPtr & m_PoolIndexBitmask) >> m_Log2PoolSize;
        uint32_t inPoolIndex = chunkPtr & m_InPoolBitmask;
        m_Pools[poolsIndex][inPoolIndex].m_Next = m_FreeChunk;
        m_FreeChunk = chunkPtr;
        n_AllocatedChunks--;
    }

    /**
     * @brief Use this to get the real pointer from the chunkPtr
     */
    T* getRealPointer(uint32_t chunkPtr){
        uint32_t poolsIndex = (chunkPtr & m_PoolIndexBitmask) >> m_Log2PoolSize;
        uint32_t inPoolIndex = chunkPtr & m_InPoolBitmask;

        return &m_Pools[poolsIndex][inPoolIndex].m_Data;
    }

    // for dumping it directly to a file

    void writeToFile(std::ofstream &file){
        file.write((char*)&n_AllocatedChunks, sizeof(n_AllocatedChunks));
        file.write((char*)&m_FreeChunk, sizeof(m_FreeChunk));
        file.write((char*)&m_NumPools, sizeof(m_NumPools));
        std::cout << "Writing PoolAllocator32b to file\n";
        std::cout << "n_AllocatedChunks: " << n_AllocatedChunks << "\n";
        std::cout << "m_FreeChunk: " << m_FreeChunk << "\n";
        std::cout << "m_NumPools: " << m_NumPools << "\n";
        for (int i = 0; i < m_NumPools; i++){
            uint32_t hwm = _poolHighWaterMark(i);
            file.write((char*)&hwm, sizeof(hwm));
            file.write((char*)m_Pools[i], sizeof(Chunk) * (hwm + 1));
        }
    }

    // how many bytes will be written to a file if you call writeToFile
    uint64_t getFileSize() const {
        uint64_t fileSize = sizeof(n_AllocatedChunks) + sizeof(m_FreeChunk) + sizeof(m_NumPools);
        for (int i = 0; i < m_NumPools; i++){
            fileSize += sizeof(uint32_t) + sizeof(Chunk) * (_poolHighWaterMark(i) + 1);
        }
        return fileSize;
    }

    void readFromFile(std::ifstream &file){
        int originalNumPools = m_NumPools;
        file.read((char*)&n_AllocatedChunks, sizeof(n_AllocatedChunks));
        file.read((char*)&m_FreeChunk, sizeof(m_FreeChunk));
        file.read((char*)&m_NumPools, sizeof(m_NumPools));
        std::cout << "Reading PoolAllocator32b from file\n";
        std::cout << "n_AllocatedChunks: " << n_AllocatedChunks << "\n";
        std::cout << "m_FreeChunk: " << m_FreeChunk << "\n";
        std::cout << "m_NumPools: " << m_NumPools << "\n";
        int poolSize = 0x1 << m_Log2PoolSize;
        for (int i = 0; i < m_NumPools; i++){
            uint32_t hwm = 0;
            file.read((char*)&hwm, sizeof(hwm));
            if(i >= originalNumPools){ //allocate new pools as needed
                m_Pools[i] = new Chunk[poolSize];
            }
            file.read((char*)m_Pools[i], sizeof(Chunk) * (hwm + 1));
            if ((int)(hwm + 1) < poolSize){
                _appendPoolFreeTail(m_Pools[i], i, hwm + 1);
            }
            if (i == m_NumPools - 1){
                m_LastPoolHighWaterMark = hwm;
            }
        }
        // clear out any extra pools
        if(originalNumPools > m_NumPools){
            for (int i = m_NumPools; i < originalNumPools; i++){
                delete[] m_Pools[i];
                m_Pools[i] = nullptr;
            }
        }
    }
    ~PoolAllocator32b(){
        if(m_MaxNumPools == 0){
            std::cout << "PoolAllocator32b not initialized\n";
            return; //not initialized 
        }
        std::cout << "Deleting PoolAllocator32b\n";
        for(int i = 0; i < m_NumPools; i++){
            delete[] m_Pools[i];
        }
        delete[] m_Pools;
    }
    
    //move constructor
    PoolAllocator32b(PoolAllocator32b &&other)
    {
        m_Pools = other.m_Pools;
        other.m_Pools = nullptr;
        m_NumPools = other.m_NumPools;
        m_MaxNumPools = other.m_MaxNumPools;
        other.m_MaxNumPools = 0;
        m_FreeChunk = other.m_FreeChunk;
        m_LastPoolHighWaterMark = other.m_LastPoolHighWaterMark;
        m_InPoolBitmask = other.m_InPoolBitmask;
        m_PoolIndexBitmask = other.m_PoolIndexBitmask;
        m_Log2PoolSize = other.m_Log2PoolSize;
    }

    //move assignment operator
    PoolAllocator32b &operator=(PoolAllocator32b &&other)
    {
        m_Pools = other.m_Pools;
        other.m_Pools = nullptr;
        m_NumPools = other.m_NumPools;
        m_MaxNumPools = other.m_MaxNumPools;
        other.m_MaxNumPools = 0;
        m_FreeChunk = other.m_FreeChunk;
        m_LastPoolHighWaterMark = other.m_LastPoolHighWaterMark;
        m_InPoolBitmask = other.m_InPoolBitmask;
        m_PoolIndexBitmask = other.m_PoolIndexBitmask;
        m_Log2PoolSize = other.m_Log2PoolSize;
        return *this;
    }

    //copy
    PoolAllocator32b(const PoolAllocator32b &r)
    {
        assert(false); // Add this
    }
};

#endif /* POOLALLOCATOR */
