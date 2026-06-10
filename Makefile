TARGET = voxelize_stl
CXX    ?= clang++

INCLUDES = -Iinclude/Voxel \
           -Iinclude/Physics \
           -Iinclude/Math \
           -Iinclude/DataStructures \
           -Iinclude/General \
           -Iextern/tinybvh

CXXFLAGS = -std=c++17 -O2 -pthread -Wall $(INCLUDES)
LDFLAGS  = -pthread

USE_OPENCL ?= 0
UNAME := $(shell uname)

ifeq ($(USE_OPENCL),1)
  CXXFLAGS += -DVOXEL_USE_OPENCL
  OPENCL_SRCS = src/Voxel/OpenCLContext.cpp src/Voxel/MeshOctreeGpu.cpp
  ifeq ($(UNAME),Darwin)
    OPENCL_INCLUDES =
    OPENCL_LDFLAGS = -framework OpenCL
  else
    OPENCL_INCLUDES = -I/opt/homebrew/opt/opencl-headers/include
    OPENCL_LDFLAGS = -lOpenCL
  endif
  CXXFLAGS += $(OPENCL_INCLUDES)
  LDFLAGS += $(OPENCL_LDFLAGS)
else
  OPENCL_SRCS =
endif

CORE_SRCS = \
  main.cpp \
  src/STLLoader.cpp \
  src/Voxel/VoxelOctree.cpp \
  src/Voxel/OctreeGeneration.cpp \
  src/Voxel/MeshOctreeGen.cpp \
  src/Voxel/TriangleBVH.cpp \
  src/Physics/AABB.cpp \
  src/Physics/GeoPrimitives.cpp \
  src/Math/cgVec.cpp \
  src/General/CumulativeProfiler.cpp \
  src/General/Stopwatch.cpp

SRCS = $(CORE_SRCS) $(OPENCL_SRCS)
OBJS = $(SRCS:.cpp=.o)

TINYBVH_URL = https://raw.githubusercontent.com/jbikker/tinybvh/main/tiny_bvh.h

.PHONY: all deps clean

all: extern/tinybvh/tiny_bvh.h $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp extern/tinybvh/tiny_bvh.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

deps: extern/tinybvh/tiny_bvh.h

extern/tinybvh/tiny_bvh.h:
	mkdir -p extern/tinybvh
	curl -fsSL $(TINYBVH_URL) -o $@

clean:
	rm -f $(OBJS) $(TARGET)
