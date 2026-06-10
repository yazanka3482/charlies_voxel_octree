#ifndef OPENCL_CONTEXT
#define OPENCL_CONTEXT

#include <cstdint>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/** Minimal OpenCL runtime used by mesh octree voxelization. */
class OpenCLContext {
public:
    static OpenCLContext& instance();

    bool isAvailable() const { return m_Available; }
    cl_context context() const { return m_Context; }
    cl_command_queue queue() const { return m_Queue; }
    cl_device_id device() const { return m_Device; }

    cl_program buildProgram(const char* source, const char* kernelName, std::string* outError = nullptr);
    cl_mem createBuffer(cl_mem_flags flags, size_t size, const void* hostPtr = nullptr);
    void writeBuffer(cl_mem buffer, size_t size, const void* data);
    void readBuffer(cl_mem buffer, size_t size, void* data);
    void release(cl_mem buffer);
    void release(cl_program program);
    void release(cl_kernel kernel);

private:
    OpenCLContext();
    ~OpenCLContext();
    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

    bool m_Available = false;
    cl_platform_id m_Platform = nullptr;
    cl_device_id m_Device = nullptr;
    cl_context m_Context = nullptr;
    cl_command_queue m_Queue = nullptr;
};

#endif
