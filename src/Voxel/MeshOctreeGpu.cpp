#include "MeshOctreeGpu.hpp"
#include "MeshOctreeGpuKernels.hpp"
#include "OpenCLContext.hpp"
#include <iostream>
#include <vector>

MeshOctreeGpu::MeshOctreeGpu(const TriangleBVH& bvh)
{
    OpenCLContext& cl = OpenCLContext::instance();
    if (!cl.isAvailable())
        return;

    const TriangleBVH::GpuLayout& layout = bvh.gpuLayout();
    if (layout.nodeCount == 0 || layout.triCount == 0)
        return;

    std::string buildLog;
    m_Program = cl.buildProgram(kMeshOctreeGpuKernelSource, "classifyVoxels", &buildLog);
    if (!m_Program)
    {
        std::cerr << "MeshOctreeGpu: OpenCL kernel build failed";
        if (!buildLog.empty())
            std::cerr << ":\n" << buildLog;
        std::cerr << std::endl;
        return;
    }

    cl_int err = CL_SUCCESS;
    m_Kernel = clCreateKernel(m_Program, "classifyVoxels", &err);
    if (!m_Kernel || err != CL_SUCCESS)
        return;

    m_BvhNodes = cl.createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        layout.nodes.size() * sizeof(TriangleBVH::GpuBvhNode), layout.nodes.data());
    m_Vertices = cl.createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        layout.vertices.size() * sizeof(tinybvh::bvhvec4), layout.vertices.data());
    m_PrimIdx = cl.createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        layout.primIdx.size() * sizeof(uint32_t), layout.primIdx.data());

    if (layout.indexed)
    {
        m_VertIdx = cl.createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            layout.vertIdx.size() * sizeof(uint32_t), layout.vertIdx.data());
        m_IndexedEnabled = 1;
    }
    else
    {
        uint32_t dummy = 0;
        m_VertIdx = cl.createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_t), &dummy);
        m_IndexedEnabled = 0;
    }

    if (!m_BvhNodes || !m_Vertices || !m_PrimIdx || !m_VertIdx)
        return;

    m_TriCount = layout.triCount;
    m_ParityBias = layout.parityRayBias;
    m_Ready = true;
}

MeshOctreeGpu::~MeshOctreeGpu()
{
    OpenCLContext& cl = OpenCLContext::instance();
    cl.release(m_BvhNodes);
    cl.release(m_Vertices);
    cl.release(m_PrimIdx);
    cl.release(m_VertIdx);
    cl.release(m_VoxelIn);
    cl.release(m_ResultOut);
    cl.release(m_Kernel);
    cl.release(m_Program);
}

void MeshOctreeGpu::evaluateBatch(const geo::aabb* voxels, OctreeVoxelGenDesc* out, size_t count)
{
    if (!m_Ready || count == 0)
        return;

    OpenCLContext& cl = OpenCLContext::instance();

    std::vector<GpuVoxelAabb> gpuVoxels(count);
    for (size_t i = 0; i < count; i++)
    {
        gpuVoxels[i].c1[0] = voxels[i].c1.x;
        gpuVoxels[i].c1[1] = voxels[i].c1.y;
        gpuVoxels[i].c1[2] = voxels[i].c1.z;
        gpuVoxels[i].pad0 = 0.f;
        gpuVoxels[i].c2[0] = voxels[i].c2.x;
        gpuVoxels[i].c2[1] = voxels[i].c2.y;
        gpuVoxels[i].c2[2] = voxels[i].c2.z;
        gpuVoxels[i].pad1 = 0.f;
    }

    if (count > m_VoxelBufferCapacity)
    {
        cl.release(m_VoxelIn);
        cl.release(m_ResultOut);
        m_VoxelIn = cl.createBuffer(CL_MEM_READ_ONLY, count * sizeof(GpuVoxelAabb));
        m_ResultOut = cl.createBuffer(CL_MEM_WRITE_ONLY, count * sizeof(GpuVoxelResult));
        m_VoxelBufferCapacity = count;
    }

    if (!m_VoxelIn || !m_ResultOut)
        return;

    cl.writeBuffer(m_VoxelIn, count * sizeof(GpuVoxelAabb), gpuVoxels.data());

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(m_Kernel, 0, sizeof(cl_mem), &m_VoxelIn);
    err |= clSetKernelArg(m_Kernel, 1, sizeof(cl_mem), &m_ResultOut);
    uint32_t count32 = (uint32_t)count;
    err |= clSetKernelArg(m_Kernel, 2, sizeof(uint32_t), &count32);
    err |= clSetKernelArg(m_Kernel, 3, sizeof(float), &m_ParityBias);
    err |= clSetKernelArg(m_Kernel, 4, sizeof(cl_mem), &m_BvhNodes);
    err |= clSetKernelArg(m_Kernel, 5, sizeof(cl_mem), &m_Vertices);
    err |= clSetKernelArg(m_Kernel, 6, sizeof(cl_mem), &m_PrimIdx);
    err |= clSetKernelArg(m_Kernel, 7, sizeof(cl_mem), &m_VertIdx);
    err |= clSetKernelArg(m_Kernel, 8, sizeof(uint32_t), &m_IndexedEnabled);
    err |= clSetKernelArg(m_Kernel, 9, sizeof(uint32_t), &m_TriCount);
    if (err != CL_SUCCESS)
        return;

    size_t global = count;
    err = clEnqueueNDRangeKernel(cl.queue(), m_Kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        return;

    std::vector<GpuVoxelResult> gpuResults(count);
    cl.readBuffer(m_ResultOut, count * sizeof(GpuVoxelResult), gpuResults.data());

    for (size_t i = 0; i < count; i++)
    {
        out[i].loc = (OctreeVolumeLocation)gpuResults[i].loc;
        out[i].type = (uint8_t)gpuResults[i].type;
        out[i].noneType = gpuResults[i].noneType != 0;
    }
}
