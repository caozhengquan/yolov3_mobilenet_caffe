cmake_minimum_required(VERSION 3.8)
project(yolov3)

set(CMAKE_CXX_STANDARD_REQUIRED 11)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)


# turn off if no cuda
option(CPU_ONLY "add cuda support or not" OFF)
option(USE_CUDNN "add cuda support or not" ON)

# -------------- ! find opencv and caffe
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv not found")
endif ()

# how to find caffe anyway?
include_directories(./include /usr/local/include/)
find_library(caffe_LIBRARY
        NAMES caffe
        HINTS "./lib"
        PATHS "./lib")

# ---------------- ! find boost
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS system)
if (NOT Boost_FOUND)
    message(ERROR "Boost not found.")
endif ()

# --------------- ! find protobuf
find_package(Protobuf REQUIRED)
if (NOT Protobuf_FOUND)
    message(ERROR "Protobuf required but not found.")
endif ()


# ---------------- ! find gflags
find_package(gflags REQUIRED)
if (NOT gflags_FOUND)
    message(ERROR "gflags required but not found.")
endif ()


# -------------- ! find thor library
find_library(thor_LIBRARY
        NAME thor
        HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        )
if (NOT thor_FOUND)
    message(ERROR "Thor lib not found, you can clone from https://github.com/jinfagang/Thor to build and install.")
endif ()


add_executable(yolo_det_trafficlight  src/yolo_det_trafficlight.cpp)
target_link_libraries(yolo_det_trafficlight ${caffe_LIBRARY} ${OpenCV_LIBS} glog ${Boost_LIBRARIES} ${Protobuf_LIBRARIES} ${thor_LIBRARY} gflags)

add_executable(yolo_det_voc  src/yolo_det_voc.cpp)
target_link_libraries(yolo_det_voc ${caffe_LIBRARY} ${OpenCV_LIBS} glog ${Boost_LIBRARIES} ${Protobuf_LIBRARIES} ${thor_LIBRARY} gflags)
if (CPU_ONLY)
    message(STATUS "build in CPU mode.")
else ()
    message(STATUS "build in CUDA mode.")
    find_package(CUDA 9.0 REQUIRED)
    if (NOT CUDA_FOUND)
        message(ERROR "CUDA 9.0 required but not found.")
    else ()
        message(STATUS "Found CUDA.")
    endif ()

    # manually set cuda include
    include_directories(/usr/local/cuda/include)
    target_link_libraries(yolo_det_trafficlight ${CUDA_LIBRARIES})
    target_link_libraries(yolo_det_voc ${CUDA_LIBRARIES})
endif ()

