cmake_minimum_required(VERSION 2.8.12)
project(nextfloor)

# pick a version #
set(nextfloor_VERSION 0.3)

include_directories(src)
set (CMAKE_CXX_STANDARD 17)
# define bin folder as destination folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(graphics_SRCS
    src/nextfloor/graphics/shape3d.cc
    src/nextfloor/graphics/quad.cc
    src/nextfloor/graphics/cube.cc
    src/nextfloor/graphics/border.cc)

set(core_SRCS
    src/nextfloor/core/config_engine.cc)

set(renderer_SRCS
    src/nextfloor/renderer/loopgl.cc)

set(physics_SRCS
    src/nextfloor/physics/collision_engine.cc
    src/nextfloor/physics/cl_collision_engine.cc
    src/nextfloor/physics/tbb_collision_engine.cc
    src/nextfloor/physics/serial_collision_engine.cc)

set(universe_SRCS
    src/nextfloor/universe/random_universe_factory.cc
    src/nextfloor/universe/camera.cc
    src/nextfloor/universe/model3d.cc
    src/nextfloor/universe/wall.cc
    src/nextfloor/universe/brick.cc
    src/nextfloor/universe/room.cc
    src/nextfloor/universe/universe.cc)

set(nextfloor_SRCS
    ${graphics_SRCS}
    ${core_SRCS}
    ${renderer_SRCS}
    ${physics_SRCS}
    ${universe_SRCS}
    src/nextfloor.cc)

set(graphics_HDRS
    src/nextfloor/graphics/shape3d.h
    src/nextfloor/graphics/quad.h
    src/nextfloor/graphics/cube.h
    src/nextfloor/graphics/border.h)

set(core_HDRS
    src/nextfloor/core/config_engine.h)

set(renderer_HDRS
    src/nextfloor/renderer/loopgl.h)

set(physics_HDRS
    src/nextfloor/physics/collision_engine.h
    src/nextfloor/physics/cl_collision_engine.h
    src/nextfloor/physics/tbb_collision_engine.h
    src/nextfloor/physics/serial_collision_engine.h)

set(universe_HDRS
    src/nextfloor/universe/random_universe_factory.h
    src/nextfloor/universe/universe_factory.h
    src/nextfloor/universe/camera.h
    src/nextfloor/universe/model3d.h
    src/nextfloor/universe/wall.h
    src/nextfloor/universe/brick.h
    src/nextfloor/universe/room.h
    src/nextfloor/universe/universe.h)

set(nextfloor_HDRS
    ${graphics_HDRS}
    ${core_HDRS}
    ${renderer_HDRS}
    ${physics_HDRS}
    ${universe_HDRS})

add_executable(nextfloor ${nextfloor_SRCS} ${nextfloor_HDRS})

find_package(TBB REQUIRED)
find_package(Config++ REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
add_definitions("-std=gnu++17")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${Config++_INCLUDE_DIRS})
target_link_libraries(nextfloor ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} config++ tbb)
target_compile_features(nextfloor PUBLIC cxx_nullptr)
