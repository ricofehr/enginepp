cmake_minimum_required(VERSION 2.8.12)
project(engine)


# pick a version #
set(engine_VERSION 0.1)

include_directories(src)
set (CMAKE_CXX_STANDARD 14)
# define bin folder as destination folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(geometry_SRCS
    src/engine/geometry/shape3d.cc
    src/engine/geometry/quad.cc
    src/engine/geometry/cube.cc
    src/engine/geometry/box.cc)

set(core_SRCS
    src/engine/core/config_engine.cc)

set(renderer_SRCS
    src/engine/renderer/loopgl.cc)

set(physics_SRCS
    src/engine/physics/collision_engine.cc
    src/engine/physics/cl_collision_engine.cc
    src/engine/physics/cilk_collision_engine.cc
    src/engine/physics/serial_collision_engine.cc)

set(universe_SRCS
    src/engine/universe/camera.cc
    src/engine/universe/model3d.cc
    src/engine/universe/wall.cc
    src/engine/universe/brick.cc
    src/engine/universe/room.cc
    src/engine/universe/universe.cc)

set(engine_SRCS
    ${geometry_SRCS}
    ${core_SRCS}
    ${renderer_SRCS}
    ${physics_SRCS}
    ${universe_SRCS}
    src/engine.cc)

set(geometry_HDRS
    src/engine/geometry/shape3d.h
    src/engine/geometry/quad.h
    src/engine/geometry/cube.h
    src/engine/geometry/box.h)

set(core_HDRS
    src/engine/core/config_engine.h)

set(renderer_HDRS
    src/engine/renderer/loopgl.h)

set(physics_HDRS
    src/engine/physics/collision_engine.h
    src/engine/physics/cl_collision_engine.h
    src/engine/physics/cilk_collision_engine.h
    src/engine/physics/serial_collision_engine.h)

set(universe_HDRS
    src/engine/universe/camera.h
    src/engine/universe/model3d.h
    src/engine/universe/wall.h
    src/engine/universe/brick.h
    src/engine/universe/room.h
    src/engine/universe/universe.h)

set(engine_HDRS
    ${geometry_HDRS}
    ${core_HDRS}
    ${renderer_HDRS}
    ${physics_HDRS}
    ${universe_HDRS})

add_executable(engine ${engine_SRCS} ${engine_HDRS})

find_package(Cilk REQUIRED)
find_package(TBB REQUIRED)
find_package(Config++ REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
add_definitions("-fcilkplus -std=gnu++14 -stdlib=libc++")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${Config++_INCLUDE_DIRS})
target_link_libraries(engine ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} config++ cilkrts tbb)
