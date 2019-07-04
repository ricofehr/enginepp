cmake_minimum_required(VERSION 2.8.12)
project(nextfloor)

# pick a version #
set(nextfloor_VERSION 0.3)

include_directories(src)
set (CMAKE_CXX_STANDARD 17)
# define bin folder as destination folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Mojave issue
IF(APPLE)
    # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
    LINK_DIRECTORIES(/usr/local/lib)
ENDIF()

set(controller_SRCS
    src/nextfloor/controller/game_loop.cc)

set(core_SRCS
    src/nextfloor/core/common_services.cc
    src/nextfloor/core/config_file.cc
    src/nextfloor/core/frame_timer.cc
    src/nextfloor/core/generic_file_io.cc
    src/nextfloor/core/program_exit.cc
    src/nextfloor/core/standard_random_generator.cc
    src/nextfloor/core/terminal_log.cc)

set(factory_SRCS
    src/nextfloor/factory/demo_universe_factory.cc)

set(objects_SRCS
    src/nextfloor/objects/border.cc
    src/nextfloor/objects/brick.cc
    src/nextfloor/objects/camera.cc
    src/nextfloor/objects/grid.cc
    src/nextfloor/objects/grid_box.cc
    src/nextfloor/objects/model.cc
    src/nextfloor/objects/room.cc
    src/nextfloor/objects/room_grid.cc
    src/nextfloor/objects/room_grid_box.cc
    src/nextfloor/objects/universe.cc
    src/nextfloor/objects/universe_grid.cc
    src/nextfloor/objects/universe_grid_box.cc
    src/nextfloor/objects/wall.cc)

set(physics_SRCS
    src/nextfloor/physics/cl_collision.cc
    src/nextfloor/physics/collision.cc
    src/nextfloor/physics/serial_collision.cc
    src/nextfloor/physics/tbb_collision.cc)

set(polygons_SRCS
    src/nextfloor/polygons/polygon.cc
    src/nextfloor/polygons/cube.cc)

set(renderer_SRCS
    src/nextfloor/renderer/fragment_shader.cc
    src/nextfloor/renderer/game_window.cc
    src/nextfloor/renderer/gl_cube_renderer.cc
    src/nextfloor/renderer/shader.cc
    src/nextfloor/renderer/vertex_shader.cc)

set(nextfloor_SRCS
    ${controller_SRCS}
    ${core_SRCS}
    ${factory_SRCS}
    ${objects_SRCS}
    ${physics_SRCS}
    ${polygons_SRCS}
    ${renderer_SRCS}
    src/nextfloor.cc)

set(controller_HDRS
    src/nextfloor/controller/game_loop.h)

set(core_HDRS
    src/nextfloor/core/common_services.h
    src/nextfloor/core/config.h
    src/nextfloor/core/exit.h
    src/nextfloor/core/config_file.h
    src/nextfloor/core/file_io.h
    src/nextfloor/core/frame_timer.h
    src/nextfloor/core/generic_file_io.h
    src/nextfloor/core/log.h
    src/nextfloor/core/program_exit.h
    src/nextfloor/core/random_generator.h
    src/nextfloor/core/standard_random_generator.h
    src/nextfloor/core/terminal_log.h
    src/nextfloor/core/timer.h)

set(factory_HDRS
    src/nextfloor/factory/demo_universe_factory.h
    src/nextfloor/factory/universe_factory.h)

set(objects_HDRS
    src/nextfloor/objects/border.h
    src/nextfloor/objects/brick.h
    src/nextfloor/objects/camera.h
    src/nextfloor/objects/engine_border.h
    src/nextfloor/objects/engine_collision.h
    src/nextfloor/objects/engine_grid.h
    src/nextfloor/objects/engine_grid_box.h
    src/nextfloor/objects/engine_object.h
    src/nextfloor/objects/engine_renderer.h
    src/nextfloor/objects/grid.h
    src/nextfloor/objects/grid_box.h
    src/nextfloor/objects/room_grid_box.h
    src/nextfloor/objects/model.h
    src/nextfloor/objects/room.h
    src/nextfloor/objects/room_grid.h
    src/nextfloor/objects/universe.h
    src/nextfloor/objects/universe_grid.h
    src/nextfloor/objects/universe_grid_box.h
    src/nextfloor/objects/wall.h)

set(physics_HDRS
    src/nextfloor/physics/cl_collision.h
    src/nextfloor/physics/collision.h
    src/nextfloor/physics/serial_collision.h
    src/nextfloor/physics/tbb_collision.h)

set(polygons_HDRS
    src/nextfloor/polygons/polygon.h
    src/nextfloor/polygons/cube.h)

set(renderer_HDRS
    src/nextfloor/renderer/fragment_shader.h
    src/nextfloor/renderer/game_window.h
    src/nextfloor/renderer/gl_cube_renderer.h
    src/nextfloor/renderer/gl_renderer.h
    src/nextfloor/renderer/shader.h
    src/nextfloor/renderer/vertex_shader.h)

set(nextfloor_HDRS
    ${controller_HDRS}
    ${core_HDRS}
    ${factory_HDRS}
    ${objects_HDRS}
    ${polygons_HDRS}
    ${renderer_HDRS})

add_executable(nextfloor ${nextfloor_SRCS} ${nextfloor_HDRS})

find_package(TBB REQUIRED)
find_package(Config++ REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
find_library(SOIL REQUIRED)
add_definitions("-std=gnu++17 -Wall")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${Config++_INCLUDE_DIRS})
target_link_libraries(nextfloor ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} config++ tbb)
target_compile_features(nextfloor PUBLIC cxx_nullptr)
