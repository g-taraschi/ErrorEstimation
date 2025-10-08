cmake_minimum_required (VERSION 3.14) # CMake minimum version required. It is always good to ensure!
project (H1Test) # name of your project

# ----- Find PZ. By default searches one directory up -----
find_package(NeoPZ REQUIRED HINTS ${CMAKE_SOURCE_DIR}/../neopz_install/ ${CMAKE_SOURCE_DIR}/neopz_install/)

add_executable(estimation2D estimation2D.cpp) 
add_executable(estimation3D estimation3D.cpp)
add_executable(href3D href3D.cpp)
add_executable(depTest dependencyTest.cpp) 

target_link_libraries(estimation2D NeoPZ::pz) 
target_link_libraries(estimation3D NeoPZ::pz)
target_link_libraries(href3D NeoPZ::pz)
target_link_libraries(depTest NeoPZ::pz) 
