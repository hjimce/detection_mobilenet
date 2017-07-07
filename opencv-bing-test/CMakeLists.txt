cmake_minimum_required(VERSION 3.0)

project(opencv-bing-test)
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("saliency/include")
INCLUDE_DIRECTORIES("saliency/src/BING")
INCLUDE_DIRECTORIES("saliency/src")

file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE saliency_src saliency/src/*.cpp )
file(GLOB_RECURSE saliency_bing saliency/src/BING/*.cpp )
add_executable(opencv_bing ${SRCS} ${saliency_src} ${saliency_bing})
target_link_libraries(opencv_bing glog ${OpenCV_LIBS})
