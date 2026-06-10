#ifndef OCTREE
#define OCTREE
#include <iostream>
#include <PoolAllocator.hpp>
#include <assert.h>
#include <stack>

struct OctreeCellDescriptor{
    int level = 0;
    bool hasChildren = false;
};

/**
 * @brief Space optimized octree that stores 1 bit per leaf node
 * 
 */
struct BitOctree{

    #pragma pack(push, 1)
    struct BranchNode{
        uint8_t m_Data=0;
        uint32_t m_Children=0; //0 indicates no children
    };

    struct EightBranchNodes {
        EightBranchNodes(){}
        BranchNode m_ChildNodes[8];
    };

    // For the nodes at the bottom of the tree we dont need to store pointers
    // this is only allowed at the lowest octree level
    struct EightLeafNodes{
        EightLeafNodes(){}
        uint8_t m_ChildData = 0;

        uint8_t get(int i) const { return (m_ChildData >> i) & 1; }
        void set(int i, uint8_t v) {
            m_ChildData = (m_ChildData & ~(uint8_t(1) << i)) | ((v & 1) << i);
        }
        void fill(uint8_t v) { m_ChildData = (v & 1) ? 0xFF : 0; }
    };

    #pragma pack(pop)

    BranchNode m_RootNode;
    PoolAllocator32b<EightBranchNodes> m_BranchNodes = PoolAllocator32b<EightBranchNodes>();
    PoolAllocator32b<EightLeafNodes> m_LeafNodes = PoolAllocator32b<EightLeafNodes>();

    int m_MaxDepth=1; //max octree depth, leaf nodes will be stored at this level

    void** _temp_node_chain = nullptr; //keeps track of the chain of nodes as we descend the octree
                                // so that we can go back up the tree and collapse nodes if needed

    BitOctree(){
        _temp_node_chain=nullptr;
        m_MaxDepth = 0;
    }

    BitOctree(int maxDepth){
        m_MaxDepth = maxDepth;
        std::cout << "Creating bit octree " << _temp_node_chain << "\n";
        _temp_node_chain = new void* [m_MaxDepth+1]; //used to keep track of child chain during data setting
        m_BranchNodes = PoolAllocator32b<EightBranchNodes>(22, 128);
        m_LeafNodes = PoolAllocator32b<EightLeafNodes>(22, 128);
    }

    ~BitOctree(){
        std::cout << "Deleting bit octree " << _temp_node_chain << "\n";
        if(_temp_node_chain != nullptr){
            delete[] _temp_node_chain;
        }
    }


    /**
     * @brief Use this to read from the octree
     * 
     * @param x 
     * @param y 
     * @param z 
     * @return uint32_t child index, or 0 if no children
     */
    inline uint8_t getDataAt(int64_t indX, int64_t indY, int64_t indZ, int maxlevel, OctreeCellDescriptor &cellDesc){
        assert(m_MaxDepth != 0);
        BranchNode currentNode = m_RootNode;
        int bit = m_MaxDepth - 1; // which bit to pull from the indices to determine the index to look at //7
                                   //  ex: 8 levels:
                                   //  10100110 ind
                                   //  76543210 level corresponding to each bit

        while (bit >= maxlevel)
        {
            int64_t mask = int64_t(0x1) << bit;
            uint8_t bitX = (indX & mask) != 0;
            uint8_t bitY = (indY & mask) != 0;
            uint8_t bitZ = (indZ & mask) != 0;
            uint32_t nextChildren = currentNode.m_Children;
            if(nextChildren == 0){ //branch has no children
                cellDesc.level = bit;
                cellDesc.hasChildren = false;
                uint8_t result = currentNode.m_Data;
                return result;
            }
            if(bit == 0) { // the children are leaf nodes at the bottom of the octree
                cellDesc.level = 0;
                cellDesc.hasChildren = false; // returns a leaf node, so no children
                EightLeafNodes* children = m_LeafNodes.getRealPointer(nextChildren);
                return children->get(bitX << 2 | bitY << 1 | bitZ);
            }
            // children are branch nodes
            EightBranchNodes* children = m_BranchNodes.getRealPointer(nextChildren);
            currentNode = children->m_ChildNodes[bitX << 2 | bitY << 1 | bitZ];
            bit--;
        }
        cellDesc.level = maxlevel;
        cellDesc.hasChildren = currentNode.m_Children != 0;
        return currentNode.m_Data;
    }

    inline void setDataAt(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data){
        BranchNode* currentNode = &m_RootNode;
        int bit = m_MaxDepth - 1;
        int node_chain_length = 0;

        while(bit >= level){
            _temp_node_chain[node_chain_length] = currentNode;
            node_chain_length++;

            // get the next node
            int64_t mask = int64_t(0x1) << bit;
            uint8_t bitX = (indX & mask) != 0;
            uint8_t bitY = (indY & mask) != 0;
            uint8_t bitZ = (indZ & mask) != 0;
            uint32_t nextChildren = currentNode->m_Children;

            if(nextChildren == 0){ //branch has no children, so create them
                if(bit == 0){
                    //at the lowest level so create leaf nodes
                    uint32_t newEightLeaf = m_LeafNodes.allocate();
                    if(newEightLeaf == 0){
                        std::cout << "Warning, out of octree pool memory\n";
                        return;
                    } //out of pool memory
                    currentNode->m_Children = newEightLeaf;
                    // this is the lowest possible node, so just set it
                    EightLeafNodes* newLeaves = m_LeafNodes.getRealPointer(newEightLeaf);
                    newLeaves->fill(currentNode->m_Data);
                    newLeaves->set(bitX << 2 | bitY << 1 | bitZ, data);
                    _temp_node_chain[node_chain_length] = newLeaves;
                    node_chain_length++;
                    break;
                } else {
                    // create branch node
                    uint32_t newEightBranch = m_BranchNodes.allocate();
                    if(newEightBranch == 0){
                        std::cout << "Warning, out of octree pool memory\n";
                        return;
                    } //out of pool memory

                    currentNode->m_Children = newEightBranch;
                    EightBranchNodes* newNodes = m_BranchNodes.getRealPointer(newEightBranch);
                    for(int i = 0; i < 8; i++){
                        newNodes->m_ChildNodes[i].m_Data = currentNode->m_Data;
                        newNodes->m_ChildNodes[i].m_Children = 0;
                    }
                    currentNode = &newNodes->m_ChildNodes[bitX << 2 | bitY << 1 | bitZ];
                }
            } else {
                if(bit == 0){
                    // reached the lowest level and there are children
                    EightLeafNodes* leaves = m_LeafNodes.getRealPointer(nextChildren);
                    leaves->set(bitX << 2 | bitY << 1 | bitZ, data);
                    _temp_node_chain[node_chain_length] = leaves;
                    node_chain_length++;
                    break;
                }
                currentNode = &m_BranchNodes.getRealPointer(nextChildren)->m_ChildNodes[bitX << 2 | bitY << 1 | bitZ];
            }
            
            if(bit == level){ // at the target node, bit will never be 0 here
                currentNode->m_Data = data & 1;
            }

            bit--;
        }

        _updateNodeChain(_temp_node_chain, node_chain_length);
    }

    inline void replaceDataAt(int64_t indX, int64_t indY, int64_t indZ, int level, uint8_t data, uint8_t dataToIgnore){
        BranchNode* currentNode = &m_RootNode;
        int bit = m_MaxDepth - 1;
        int node_chain_length = 0;

        while(bit >= level){
            _temp_node_chain[node_chain_length] = currentNode;
            node_chain_length++;

            // get the next node
            int64_t mask = int64_t(0x1) << bit;
            uint8_t bitX = (indX & mask) != 0;
            uint8_t bitY = (indY & mask) != 0;
            uint8_t bitZ = (indZ & mask) != 0;
            uint32_t nextChildren = currentNode->m_Children;

            if(nextChildren == 0){ //branch has no children, so create them
                if(currentNode->m_Data == dataToIgnore){
                    return; //cant modify this node, the resulting children will have dataToIgnore
                }
                if(bit == 0){
                    //at the lowest level so create leaf nodes
                    uint32_t newEightLeaf = m_LeafNodes.allocate();
                    if(newEightLeaf == 0){
                        std::cout << "Warning, out of octree pool memory\n";
                        return;
                    } //out of pool memory
                    currentNode->m_Children = newEightLeaf;
                    // this is the lowest possible node, so just set it
                    EightLeafNodes* newLeaves = m_LeafNodes.getRealPointer(newEightLeaf);
                    newLeaves->fill(currentNode->m_Data);
                    newLeaves->set(bitX << 2 | bitY << 1 | bitZ, data);
                    _temp_node_chain[node_chain_length] = newLeaves;
                    node_chain_length++;
                    break;
                } else {
                    // create branch node
                    uint32_t newEightBranch = m_BranchNodes.allocate();
                    if(newEightBranch == 0){
                        std::cout << "Warning, out of octree pool memory\n";
                        return;
                    } //out of pool memory
                    currentNode->m_Children = newEightBranch;
                    EightBranchNodes* newNodes = m_BranchNodes.getRealPointer(newEightBranch);
                    for(int i = 0; i < 8; i++){
                        newNodes->m_ChildNodes[i].m_Data = currentNode->m_Data;
                        newNodes->m_ChildNodes[i].m_Children = 0;
                    }
                    currentNode = &newNodes->m_ChildNodes[bitX << 2 | bitY << 1 | bitZ];
                }
            } else {
                if(bit == 0){
                    // reached the lowest level and there are children
                    EightLeafNodes* leaves = m_LeafNodes.getRealPointer(nextChildren);
                    leaves->set(bitX << 2 | bitY << 1 | bitZ, data);
                    _temp_node_chain[node_chain_length] = leaves;
                    node_chain_length++;
                    break;
                }
                currentNode = &m_BranchNodes.getRealPointer(nextChildren)->m_ChildNodes[bitX << 2 | bitY << 1 | bitZ];
            }
            
            if(bit == level){ // at the target node, bit will never be 0 here

                // set current node data
                if(currentNode->m_Data != dataToIgnore){
                    currentNode->m_Data = data & 1;
                }

                // the current node has children, so we need to traverse the lower branches and set all of them
                if(currentNode->m_Children != 0){
                    if(bit > 1){ // children arent leaf nodes, have to traverse the lower branches and set all of them
                        //branch nodes, the level of the branch nodes
                        std::stack<std::tuple<EightBranchNodes*, int>> childrenToVisit;
                        childrenToVisit.push(std::make_tuple(m_BranchNodes.getRealPointer(currentNode->m_Children), bit-1));

                        while(!childrenToVisit.empty()){
                            EightBranchNodes* currentBranches = std::get<0>(childrenToVisit.top());
                            int currentLevel = std::get<1>(childrenToVisit.top());
                            childrenToVisit.pop();

                            //set the data for each child
                            for(int i = 0; i < 8; i++){
                                if(currentBranches->m_ChildNodes[i].m_Data != dataToIgnore){
                                    currentBranches->m_ChildNodes[i].m_Data = data & 1;
                                }
                            }

                            //handle the children of the children
                            if(currentLevel == 1) { //children of the children are leaf nodes
                                for(int i = 0; i < 8; i++){
                                    if(currentBranches->m_ChildNodes[i].m_Children != 0){ //has leaf nodes
                                        EightLeafNodes* leaves = m_LeafNodes.getRealPointer(currentBranches->m_ChildNodes[i].m_Children);
                                        for(int j = 0; j < 8; j++){
                                            if(leaves->get(j) != dataToIgnore){
                                                leaves->set(j, data);
                                            }
                                        }
                                    }
                                }
                            } else { //children of the children are branch nodes, add them to the stack
                                for(int i = 0; i < 8; i++){
                                    if(currentBranches->m_ChildNodes[i].m_Children != 0){
                                        childrenToVisit.push(std::make_tuple(m_BranchNodes.getRealPointer(currentBranches->m_ChildNodes[i].m_Children), currentLevel-1));
                                    }
                                }
                            }

                        }
                    } else { // children are leaf nodes so easy to just set them
                        EightLeafNodes* leaves = m_LeafNodes.getRealPointer(currentNode->m_Children);
                        for(int i = 0; i < 8; i++){
                            if(leaves->get(i) != dataToIgnore){
                                leaves->set(i, data);
                            }
                        }
                    }
                }
                break; // break because we are at the target node
            }

            bit--;
        }

        _updateNodeChain(_temp_node_chain, node_chain_length);
    }

    // goes up the chain and collapses nodes where all the data matches
    inline void _updateNodeChain(void** node_chain, int node_chain_length){
        // go up node chain to recursively update/collapse nodes where all the data matches
        for (int i = node_chain_length-1; i>=1; i--){ //skip the root node since it doesnt have a parent
            if(i == m_MaxDepth){ 
                EightLeafNodes* leafNodes = (EightLeafNodes*)node_chain[i];
                bool dataIsAllTheSame = true;
                bool foundMostCommon = false;
                uint8_t mostCommonData = 0;
                uint8_t first = leafNodes->get(0);
                for(int j = 1; j < 8; j++){
                    if(leafNodes->get(j) != first){
                        dataIsAllTheSame = false;
                        break;
                    }
                }
                if(!dataIsAllTheSame){
                    // find the most common data
                    uint8_t nMatches[8] = {0};
                    for(int j = 0; j < 8; j++){
                        for(int k = 0; k < 8; k++){
                            if(leafNodes->get(j) == leafNodes->get(k)){
                                nMatches[j]++;
                            }
                        }
                    }
                    mostCommonData = leafNodes->get(0);
                    int maxMatches = nMatches[0];
                    for(int j = 1; j < 8; j++){
                        if(nMatches[j] > maxMatches){
                            maxMatches = nMatches[j];
                            mostCommonData = leafNodes->get(j);
                        }
                    }
                    if(maxMatches >= 4){ // only run if there are at least 4 children with the same data
                        foundMostCommon = true;
                    }
                }
                if(dataIsAllTheSame){ //collapse the leaf nodes into the parent node
                    uint8_t data = first;
                    BranchNode* parent = (BranchNode*)node_chain[i-1];
                    uint32_t leafPtr = parent->m_Children;
                    m_LeafNodes.deallocate(leafPtr);
                    parent->m_Children=0;
                    parent->m_Data = data;
                } else if (foundMostCommon){
                    BranchNode* parent = (BranchNode*)node_chain[i-1];
                    if(parent->m_Data != mostCommonData){
                        parent->m_Data = mostCommonData;
                    } else {
                        break; // no parent modification, stop going up the chain
                    }
                } else {
                    break; //if we didnt modify the parent, there is no need to continue going up the chain
                }
            } else {
                BranchNode* parent = (BranchNode*)node_chain[i-1];
                EightBranchNodes* children = m_BranchNodes.getRealPointer(parent->m_Children);
                bool dataIsAllTheSame = false;
                bool foundMostCommon = false;
                uint8_t mostCommonData = 0;

                if(children->m_ChildNodes[1].m_Data == children->m_ChildNodes[0].m_Data &&
                   children->m_ChildNodes[2].m_Data == children->m_ChildNodes[0].m_Data && 
                   children->m_ChildNodes[3].m_Data == children->m_ChildNodes[0].m_Data && 
                   children->m_ChildNodes[4].m_Data == children->m_ChildNodes[0].m_Data && 
                   children->m_ChildNodes[5].m_Data == children->m_ChildNodes[0].m_Data && 
                   children->m_ChildNodes[6].m_Data == children->m_ChildNodes[0].m_Data && 
                   children->m_ChildNodes[7].m_Data == children->m_ChildNodes[0].m_Data ){
                    dataIsAllTheSame = true;
                } else {
                    // find the most common data
                    uint8_t nMatches[8] = {0};
                    for(int j = 0; j < 8; j++){
                        for(int k = 0; k < 8; k++){
                            if(children->m_ChildNodes[j].m_Data == children->m_ChildNodes[k].m_Data){
                                nMatches[j]++;
                            }
                        }
                    }
                    mostCommonData = children->m_ChildNodes[0].m_Data;
                    int maxMatches = nMatches[0];
                    for(int j = 1; j < 8; j++){
                        if(nMatches[j] > maxMatches){
                            maxMatches = nMatches[j];
                            mostCommonData = children->m_ChildNodes[j].m_Data;
                        }
                    }
                    if(maxMatches >= 4){ // only run if there are at least 4 children with the same data
                        foundMostCommon = true;
                    }
                }

                if(dataIsAllTheSame){
                    //check if all the children dont have children
                    bool areLeafNodes = false;
                    if(children->m_ChildNodes[0].m_Children == 0 &&
                       children->m_ChildNodes[1].m_Children == 0 &&
                       children->m_ChildNodes[2].m_Children == 0 &&
                       children->m_ChildNodes[3].m_Children == 0 &&
                       children->m_ChildNodes[4].m_Children == 0 &&
                       children->m_ChildNodes[5].m_Children == 0 &&
                       children->m_ChildNodes[6].m_Children == 0 &&
                       children->m_ChildNodes[7].m_Children == 0)
                       {
                            areLeafNodes = true;
                       }
                    if(areLeafNodes){
                        // we can destroy these children
                        uint8_t data = children->m_ChildNodes[0].m_Data;
                        m_BranchNodes.deallocate(parent->m_Children);
                        parent->m_Children = 0;
                        parent->m_Data = data;
                    } else {
                        break; //no modifications to this node
                    }
                } else if(foundMostCommon){
                    // set this node to the most common data
                    parent->m_Data = mostCommonData;
                } else {
                    break; //no modifications to this node
                }
            }
        }
    }

    void writeToFile(std::ofstream &file){
        file.write((char*)&m_MaxDepth, sizeof(m_MaxDepth));
        file.write((char*)&m_RootNode, sizeof(m_RootNode));
        m_BranchNodes.writeToFile(file);
        m_LeafNodes.writeToFile(file);
    }

    void readFromFile(std::ifstream &file){
        std::cout << "Reading BitOctree from file\n";
        int originalMaxDepth = m_MaxDepth;
        file.read((char*)&m_MaxDepth, sizeof(m_MaxDepth));
        file.read((char*)&m_RootNode, sizeof(m_RootNode));
        if(m_MaxDepth != originalMaxDepth){
            delete[] _temp_node_chain;
            _temp_node_chain = new void* [m_MaxDepth+1];
        }
        m_BranchNodes.readFromFile(file);
        m_LeafNodes.readFromFile(file);
        //done 
    }

    //move constructor
    BitOctree(BitOctree &&other)
    {
        m_RootNode = other.m_RootNode;
        m_BranchNodes = std::move(other.m_BranchNodes);
        m_LeafNodes = std::move(other.m_LeafNodes);
        m_MaxDepth = other.m_MaxDepth;
        other.m_MaxDepth = 0;
        _temp_node_chain = other._temp_node_chain;
        other._temp_node_chain = nullptr;
    }

    //move assignment operator
    BitOctree &operator=(BitOctree &&other)
    {
        m_RootNode = other.m_RootNode;
        m_BranchNodes = std::move(other.m_BranchNodes);
        m_LeafNodes = std::move(other.m_LeafNodes);
        m_MaxDepth = other.m_MaxDepth;
        other.m_MaxDepth = 0;
        _temp_node_chain = other._temp_node_chain;
        other._temp_node_chain = nullptr;
        return *this;
    }

    //copy
    BitOctree(const BitOctree &r)
    {
        assert(false); // dont copy
    }
};

#endif /* OCTREE */
