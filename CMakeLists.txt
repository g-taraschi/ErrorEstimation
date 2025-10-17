cmake_minimum_required (VERSION 3.14) # CMake minimum version required. It is always good to ensure!

project (ErrorEstimation)

# Find PZ. By default searches one directory up
find_package(NeoPZ REQUIRED HINTS ${CMAKE_SOURCE_DIR}/../neopz_install/ ${CMAKE_SOURCE_DIR}/neopz_install/)

# Add Utils subdirectory to build UTILS_lib
add_subdirectory(Utils)

add_executable(estimation2D estimation2D.cpp) 
add_executable(estimation3D estimation3D.cpp)
add_executable(depTest dependencyTest.cpp)
add_executable(pragerSynge pragerSynge.cpp)
add_executable(GOMixed goalOrientedMixed.cpp)
add_executable(GOH1 goalOrientedH1.cpp)

target_link_libraries(estimation2D NeoPZ::pz) 
target_link_libraries(estimation3D NeoPZ::pz)
target_link_libraries(depTest NeoPZ::pz)
target_link_libraries(pragerSynge NeoPZ::pz UTILS_lib)
target_link_libraries(GOMixed NeoPZ::pz UTILS_lib)
target_link_libraries(GOH1 NeoPZ::pz UTILS_lib)
