
# Create UTILS_lib static library from all source files in this folder
file(GLOB UTILS_SRC CONFIGURE_DEPENDS "*.cpp" "*.c" "*.hpp")
file(GLOB UTILS_HEADERS CONFIGURE_DEPENDS "*.h")

add_library(UTILS_lib STATIC ${UTILS_SRC} ${UTILS_HEADERS})
target_include_directories(UTILS_lib PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../neopz_install/pz/include
)
# If NeoPZ::pz is available as an imported target, link to it
find_package(NeoPZ REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../neopz_install/)
target_link_libraries(UTILS_lib PUBLIC NeoPZ::pz)