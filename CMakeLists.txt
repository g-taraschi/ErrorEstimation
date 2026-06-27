cmake_minimum_required (VERSION 3.14) # CMake minimum version required. It is always good to ensure!

project (ErrorEstimation)

# Find PZ. By default searches one directory up
find_package(NeoPZ REQUIRED HINTS ${CMAKE_SOURCE_DIR}/../neopz_install/ ${CMAKE_SOURCE_DIR}/neopz_install/)

# Add Utils subdirectory to build UTILS_lib
add_subdirectory(Utils)

add_executable(PS PragerSynge.cpp)
add_executable(GOM GoalOrientedMixed.cpp)
add_executable(G2D GoalOrientedWell2D.cpp)

target_link_libraries(PS NeoPZ::pz UTILS_lib)
target_link_libraries(GOM NeoPZ::pz UTILS_lib)
target_link_libraries(G2D NeoPZ::pz UTILS_lib)
