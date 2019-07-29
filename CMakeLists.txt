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
        src/nextfloor/core/file_config_parser.cc
        src/nextfloor/core/frame_timer.cc
        src/nextfloor/core/std_file_io.cc
        src/nextfloor/core/program_exit.cc
        src/nextfloor/core/pseudo_random_generator.cc
        src/nextfloor/core/terminal_log.cc)

set(factory_SRCS
        src/nextfloor/factory/game_factory.cc
        src/nextfloor/factory/gl_renderer_factory.cc
        src/nextfloor/factory/services_factory.cc)

set(gameplay_SRCS
        src/nextfloor/gameplay/demo_level.cc
        src/nextfloor/gameplay/random_level.cc)

set(grid_SRCS
        src/nextfloor/grid/wired_grid.cc
        src/nextfloor/grid/wired_grid_box.cc
        src/nextfloor/grid/mesh_grid.cc
        src/nextfloor/grid/room_grid.cc
        src/nextfloor/grid/room_grid_box.cc
        src/nextfloor/grid/universe_grid.cc
        src/nextfloor/grid/universe_grid_box.cc)

set(objects_SRCS
        src/nextfloor/objects/back_wall.cc
        src/nextfloor/objects/head_camera.cc
        src/nextfloor/objects/depth_wall.cc
        src/nextfloor/objects/floor.cc
        src/nextfloor/objects/front_wall.cc
        src/nextfloor/objects/left_wall.cc
        src/nextfloor/objects/model_mesh.cc
        src/nextfloor/objects/player.cc
        src/nextfloor/objects/right_wall.cc
        src/nextfloor/objects/rock.cc
        src/nextfloor/objects/roof.cc
        src/nextfloor/objects/room.cc
        src/nextfloor/objects/universe.cc
        src/nextfloor/objects/wall.cc
        src/nextfloor/objects/wall_brick.cc
        src/nextfloor/objects/width_wall.cc)

set(physics_SRCS
        src/nextfloor/physics/cube_border.cc
        src/nextfloor/physics/cl_nearer_collision_engine.cc
        src/nextfloor/physics/nearer_collision_engine.cc
        src/nextfloor/physics/serial_nearer_collision_engine.cc
        src/nextfloor/physics/tbb_nearer_collision_engine.cc)

set(polygons_SRCS
        src/nextfloor/polygons/mesh_polygon.cc
        src/nextfloor/polygons/cube.cc)

set(renderer_SRCS
        src/nextfloor/renderer/fragment_shader.cc
        src/nextfloor/renderer/game_window.cc
        src/nextfloor/renderer/gl_cube_renderer_engine.cc
        src/nextfloor/renderer/shader.cc
        src/nextfloor/renderer/vertex_shader.cc)

set(nextfloor_SRCS
        ${controller_SRCS}
        ${core_SRCS}
        ${factory_SRCS}
        ${gameplay_SRCS}
        ${grid_SRCS}
        ${objects_SRCS}
        ${physics_SRCS}
        ${polygons_SRCS}
        ${renderer_SRCS}
        src/nextfloor.cc)

set(controller_HDRS
        src/nextfloor/controller/game_loop.h)

set(core_HDRS
        src/nextfloor/core/common_services.h
        src/nextfloor/core/config_parser.h
        src/nextfloor/core/exit.h
        src/nextfloor/core/file_config_parser.h
        src/nextfloor/core/file_io.h
        src/nextfloor/core/frame_timer.h
        src/nextfloor/core/std_file_io.h
        src/nextfloor/core/log.h
        src/nextfloor/core/program_exit.h
        src/nextfloor/core/random_generator.h
        src/nextfloor/core/pseudo_random_generator.h
        src/nextfloor/core/terminal_log.h
        src/nextfloor/core/timer.h)

set(factory_HDRS
        src/nextfloor/factory/core_factory.h
        src/nextfloor/factory/mesh_factory.h
        src/nextfloor/factory/renderer_factory.h
        src/nextfloor/factory/game_factory.h
        src/nextfloor/factory/gl_renderer_factory.h)

set(gameplay_HDRS
        src/nextfloor/gameplay/demo_level.h
        src/nextfloor/gameplay/level.h
        src/nextfloor/gameplay/random_level.h)

set(grid_HDRS
        src/nextfloor/grid/wired_grid.h
        src/nextfloor/grid/wired_grid_box.h
        src/nextfloor/grid/mesh_grid.h
        src/nextfloor/grid/room_grid.h
        src/nextfloor/grid/room_grid_box.h
        src/nextfloor/grid/universe_grid.h
        src/nextfloor/grid/universe_grid_box.h)

set(objects_HDRS
        src/nextfloor/objects/back_wall.h
        src/nextfloor/objects/head_camera.h
        src/nextfloor/objects/depth_wall.h
        src/nextfloor/objects/floor.h
        src/nextfloor/objects/front_wall.h
        src/nextfloor/objects/border.h
        src/nextfloor/objects/collision_engine.h
        src/nextfloor/objects/grid.h
        src/nextfloor/objects/grid_box.h
        src/nextfloor/objects/mesh.h
        src/nextfloor/objects/polygon.h
        src/nextfloor/objects/renderer_engine.h
        src/nextfloor/objects/left_wall.h
        src/nextfloor/objects/model_mesh.h
        src/nextfloor/objects/player.h
        src/nextfloor/objects/right_wall.h
        src/nextfloor/objects/rock.h
        src/nextfloor/objects/roof.h
        src/nextfloor/objects/room.h
        src/nextfloor/objects/universe.h
        src/nextfloor/objects/wall.h
        src/nextfloor/objects/wall_brick.h
        src/nextfloor/objects/width_wall.h)

set(physics_HDRS
        src/nextfloor/physics/cube_border.h
        src/nextfloor/physics/cl_nearer_collision_engine.h
        src/nextfloor/physics/nearer_collision_engine.h
        src/nextfloor/physics/serial_nearer_collision_engine.h
        src/nextfloor/physics/tbb_nearer_collision_engine.h)

set(polygons_HDRS
        src/nextfloor/polygons/mesh_polygon.h
        src/nextfloor/polygons/cube.h)

set(renderer_HDRS
        src/nextfloor/renderer/fragment_shader.h
        src/nextfloor/renderer/game_window.h
        src/nextfloor/renderer/gl_cube_renderer_engine.h
        src/nextfloor/renderer/gl_renderer_engine.h
        src/nextfloor/renderer/scene_window.h
        src/nextfloor/renderer/shader.h
        src/nextfloor/renderer/vertex_shader.h)

set(nextfloor_HDRS
        ${controller_HDRS}
        ${core_HDRS}
        ${factory_HDRS}
        ${gameplay_HDRS}
        ${grid_HDRS}
        ${objects_HDRS}
        ${physics_HDRS}
        ${polygons_HDRS}
        ${renderer_HDRS} src/nextfloor/factory/randomed_game_factory.cc src/nextfloor/factory/randomed_game_factory.h)

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
