cmake_minimum_required(VERSION 2.8.12)
project(engine)


# pick a version #
set(engine_VERSION 0.1)

include_directories(src)
set (CMAKE_CXX_STANDARD 14)
# define bin folder as destination folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(geometry_SRCS
    src/engine/geometry/shape3d.cc
    src/engine/geometry/quad.cc
    src/engine/geometry/cube.cc
    src/engine/geometry/box.cc)

set(helpers_SRCS
    src/engine/helpers/proxygl.cc)

set(parallell_SRCS
    src/engine/parallell/cl_parallell.cc
    src/engine/parallell/cilk_parallell.cc
    src/engine/parallell/serial_parallell.cc)

set(universe_SRCS
    src/engine/universe/camera.cc
    src/engine/universe/model3d.cc
    src/engine/universe/wall.cc
    src/engine/universe/window_model.cc
    src/engine/universe/door.cc
    src/engine/universe/brick.cc
    src/engine/universe/room.cc
    src/engine/universe/universe.cc)

set(engine_SRCS
    ${geometry_SRCS}
    ${helpers_SRCS}
    ${parallell_SRCS}
    ${universe_SRCS}
    src/engine.cc)

set(geometry_HDRS
    src/engine/geometry/shape3d.h
    src/engine/geometry/quad.h
    src/engine/geometry/cube.h
    src/engine/geometry/box.h)

set(helpers_HDRS
    src/engine/helpers/proxygl.h)

set(parallell_HDRS
    src/engine/parallell/engine_parallell.h
    src/engine/parallell/cl_parallell.h
    src/engine/parallell/cilk_parallell.h
    src/engine/parallell/serial_parallell.h)

set(universe_HDRS
    src/engine/universe/camera.h
    src/engine/universe/model3d.h
    src/engine/universe/wall.h
    src/engine/universe/window_model.h
    src/engine/universe/door.h
    src/engine/universe/brick.h
    src/engine/universe/room.h
    src/engine/universe/universe.h)

set(engine_HDRS
    ${geometry_HDRS}
    ${helpers_HDRS}
    ${parallell_HDRS}
    ${universe_HDRS})

add_executable(engine ${engine_SRCS} ${engine_HDRS})

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
add_definitions("-fcilkplus -std=gnu++14 -stdlib=libc++")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(engine ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} cilkrts tbb)
