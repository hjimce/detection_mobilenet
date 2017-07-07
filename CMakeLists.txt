cmake_minimum_required(VERSION 3.5)
project(mobilenet)
#find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(./mxnet/include/mxnet)



find_package( Threads )
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_OPENMP OFF)
add_executable(imagenet mobilenent.cpp )
#add_subdirectory("./mobilenet/mobilenet-mxnet/mxnet")
add_subdirectory(opencv-bing-test)


target_link_libraries(imagenet ${OpenCV_LIBS} ${CMAKE_CURRENT_SOURCE_DIR}/libmxnet.so)#链接mxnet库