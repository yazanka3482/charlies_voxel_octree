#include "OpenCLContext.hpp"
#include <iostream>

OpenCLContext& OpenCLContext::instance()
{
    static OpenCLContext ctx;
    return ctx;
}

OpenCLContext::OpenCLContext()
{
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0)
        return;

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS)
        return;

    cl_int err = CL_SUCCESS;
    for (cl_platform_id platform : platforms)
    {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_Device, &numDevices) == CL_SUCCESS && numDevices > 0)
        {
            m_Platform = platform;
            break;
        }
    }

    if (!m_Device)
    {
        for (cl_platform_id platform : platforms)
        {
            cl_uint numDevices = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_Device, &numDevices) == CL_SUCCESS && numDevices > 0)
            {
                m_Platform = platform;
                break;
            }
        }
    }

    if (!m_Device)
        return;

    m_Context = clCreateContext(nullptr, 1, &m_Device, nullptr, nullptr, &err);
    if (!m_Context || err != CL_SUCCESS)
        return;

#ifdef CL_VERSION_2_0
    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    m_Queue = clCreateCommandQueueWithProperties(m_Context, m_Device, props, &err);
#else
    m_Queue = clCreateCommandQueue(m_Context, m_Device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (!m_Queue || err != CL_SUCCESS)
        return;

    m_Available = true;

    char name[256] = {};
    clGetDeviceInfo(m_Device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "OpenCL initialized: " << name << std::endl;
}

OpenCLContext::~OpenCLContext()
{
    if (m_Queue)
        clReleaseCommandQueue(m_Queue);
    if (m_Context)
        clReleaseContext(m_Context);
}

cl_program OpenCLContext::buildProgram(const char* source, const char* kernelName, std::string* outError)
{
    if (!m_Available)
        return nullptr;

    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(m_Context, 1, &source, nullptr, &err);
    if (!program || err != CL_SUCCESS)
        return nullptr;

    err = clBuildProgram(program, 1, &m_Device, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, m_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, m_Device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        if (outError)
            *outError = log;
        else
            std::cerr << "OpenCL build failed for " << kernelName << ":\n" << log << std::endl;
        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

cl_mem OpenCLContext::createBuffer(cl_mem_flags flags, size_t size, const void* hostPtr)
{
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(m_Context, flags, size, const_cast<void*>(hostPtr), &err);
    return (buffer && err == CL_SUCCESS) ? buffer : nullptr;
}

void OpenCLContext::writeBuffer(cl_mem buffer, size_t size, const void* data)
{
    clEnqueueWriteBuffer(m_Queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
}

void OpenCLContext::readBuffer(cl_mem buffer, size_t size, void* data)
{
    clEnqueueReadBuffer(m_Queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
}

void OpenCLContext::release(cl_mem buffer)
{
    if (buffer)
        clReleaseMemObject(buffer);
}

void OpenCLContext::release(cl_program program)
{
    if (program)
        clReleaseProgram(program);
}

void OpenCLContext::release(cl_kernel kernel)
{
    if (kernel)
        clReleaseKernel(kernel);
}
