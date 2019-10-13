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

set(actions_SRCS
        src/nextfloor/actions/fire_action.cc
        src/nextfloor/actions/jump_action.cc
        src/nextfloor/actions/movedown_action.cc
        src/nextfloor/actions/moveleft_action.cc
        src/nextfloor/actions/moveright_action.cc
        src/nextfloor/actions/moveup_action.cc
        src/nextfloor/actions/run_action.cc
        src/nextfloor/actions/sprite_action_factory.cc)

set(core_SRCS
        src/nextfloor/core/common_services.cc
        src/nextfloor/core/file_config_parser.cc
        src/nextfloor/core/game_file_io.cc
        src/nextfloor/core/program_exit.cc
        src/nextfloor/core/pseudo_random_generator.cc
        src/nextfloor/core/services_core_factory.cc
        src/nextfloor/core/terminal_log.cc)

set(gameplay_SRCS
        src/nextfloor/gameplay/game_character_factory.cc
        src/nextfloor/gameplay/demo_game_factory.cc
        src/nextfloor/gameplay/game_level.cc
        src/nextfloor/gameplay/game_loop.cc
        src/nextfloor/gameplay/game_timer.cc
        src/nextfloor/gameplay/head_camera.cc
        src/nextfloor/gameplay/player.cc)

set(grid_SRCS
        src/nextfloor/grid/mesh_grid_factory.cc
        src/nextfloor/grid/room_grid.cc
        src/nextfloor/grid/universe_grid.cc
        src/nextfloor/grid/wired_grid.cc
        src/nextfloor/grid/wired_grid_box.cc)

set(hid_SRCS
        src/nextfloor/hid/game_input_handler.cc
        src/nextfloor/hid/mouse_hid_factory.cc
        src/nextfloor/hid/mouse_keyboard.cc)

set(objects_SRCS
        src/nextfloor/objects/back_wall.cc
        src/nextfloor/objects/depth_wall.cc
        src/nextfloor/objects/floor.cc
        src/nextfloor/objects/front_wall.cc
        src/nextfloor/objects/left_wall.cc
        src/nextfloor/objects/model_mesh.cc
        src/nextfloor/objects/model_mesh_factory.cc
        src/nextfloor/objects/right_wall.cc
        src/nextfloor/objects/rock.cc
        src/nextfloor/objects/roof.cc
        src/nextfloor/objects/room.cc
        src/nextfloor/objects/universe.cc
        src/nextfloor/objects/wall.cc
        src/nextfloor/objects/wall_brick.cc
        src/nextfloor/objects/width_wall.cc)

set(physics_SRCS
        src/nextfloor/physics/cl_nearer_collision_engine.cc
        src/nextfloor/physics/cube_border.cc
        src/nextfloor/physics/mesh_physic_factory.cc
        src/nextfloor/physics/nearer_collision_engine.cc
        src/nextfloor/physics/serial_nearer_collision_engine.cc
        src/nextfloor/physics/tbb_nearer_collision_engine.cc)

set(polygons_SRCS
        src/nextfloor/polygons/cube.cc
        src/nextfloor/polygons/mesh_polygon.cc
        src/nextfloor/polygons/mesh_polygon_factory.cc)

set(renderer_SRCS
        src/nextfloor/renderer/cube_gl_renderer_engine.cc
        src/nextfloor/renderer/fragment_gl_shader.cc
        src/nextfloor/renderer/gl_renderer_engine.cc
        src/nextfloor/renderer/gl_renderer_factory.cc
        src/nextfloor/renderer/gl_scene_input.cc
        src/nextfloor/renderer/gl_scene_window.cc
        src/nextfloor/renderer/gl_shader.cc
        src/nextfloor/renderer/gl_shader_factory.cc
        src/nextfloor/renderer/vertex_gl_shader.cc)

set(nextfloor_SRCS
        ${actions_SRCS}
        ${core_SRCS}
        ${factory_SRCS}
        ${gameplay_SRCS}
        ${grid_SRCS}
        ${hid_SRCS}
        ${objects_SRCS}
        ${physics_SRCS}
        ${polygons_SRCS}
        ${renderer_SRCS}
        src/nextfloor.cc)

set(actions_HDRS
        src/nextfloor/actions/fire_action.h
        src/nextfloor/actions/jump_action.h
        src/nextfloor/actions/movedown_action.h
        src/nextfloor/actions/moveleft_action.h
        src/nextfloor/actions/moveright_action.h
        src/nextfloor/actions/moveup_action.h
        src/nextfloor/actions/run_action.h
        src/nextfloor/actions/sprite_action_factory.h)

set(core_HDRS
        src/nextfloor/core/common_services.h
        src/nextfloor/core/config_parser.h
        src/nextfloor/core/core_factory.h
        src/nextfloor/core/exit.h
        src/nextfloor/core/file_config_parser.h
        src/nextfloor/core/file_io.h
        src/nextfloor/core/game_file_io.h
        src/nextfloor/core/log.h
        src/nextfloor/core/program_exit.h
        src/nextfloor/core/pseudo_random_generator.h
        src/nextfloor/core/random_generator.h
        src/nextfloor/core/services_core_factory.h
        src/nextfloor/core/terminal_log.h)

set(gameplay_HDRS
        src/nextfloor/gameplay/action.h
        src/nextfloor/gameplay/action_factory.h
        src/nextfloor/gameplay/demo_game_factory.h
        src/nextfloor/gameplay/frame_timer.h
        src/nextfloor/gameplay/game_character_factory.h
        src/nextfloor/gameplay/game_factory.h
        src/nextfloor/gameplay/game_level.h
        src/nextfloor/gameplay/game_loop.h
        src/nextfloor/gameplay/game_timer.h
        src/nextfloor/gameplay/head_camera.h
        src/nextfloor/gameplay/hid.h
        src/nextfloor/gameplay/hid_factory.h
        src/nextfloor/gameplay/input_handler.h
        src/nextfloor/gameplay/level.h
        src/nextfloor/gameplay/loop.h
        src/nextfloor/gameplay/player.h
        src/nextfloor/gameplay/renderer_engine.h
        src/nextfloor/gameplay/renderer_factory.h
        src/nextfloor/gameplay/scene_input.h
        src/nextfloor/gameplay/scene_window.h)

set(grid_HDRS
        src/nextfloor/grid/mesh_grid_factory.h
        src/nextfloor/grid/room_grid.h
        src/nextfloor/grid/universe_grid.h
        src/nextfloor/grid/wired_grid.h
        src/nextfloor/grid/wired_grid_box.h)

set(hid_HDRS
        src/nextfloor/hid/game_input_handler.h
        src/nextfloor/hid/mouse_hid_factory.h
        src/nextfloor/hid/mouse_keyboard.h)

set(objects_HDRS
        src/nextfloor/objects/back_wall.h
        src/nextfloor/objects/border.h
        src/nextfloor/objects/collision_engine.h
        src/nextfloor/objects/depth_wall.h
        src/nextfloor/objects/floor.h
        src/nextfloor/objects/front_wall.h
        src/nextfloor/objects/grid.h
        src/nextfloor/objects/grid_box.h
        src/nextfloor/objects/grid_factory.h
        src/nextfloor/objects/left_wall.h
        src/nextfloor/objects/mesh.h
        src/nextfloor/objects/mesh_factory.h
        src/nextfloor/objects/model_mesh.h
        src/nextfloor/objects/model_mesh_factory.h
        src/nextfloor/objects/physic_factory.h
        src/nextfloor/objects/polygon.h
        src/nextfloor/objects/polygon_factory.h
        src/nextfloor/objects/right_wall.h
        src/nextfloor/objects/rock.h
        src/nextfloor/objects/roof.h
        src/nextfloor/objects/room.h
        src/nextfloor/objects/universe.h
        src/nextfloor/objects/wall.h
        src/nextfloor/objects/wall_brick.h
        src/nextfloor/objects/width_wall.h)

set(physics_HDRS
        src/nextfloor/physics/cl_nearer_collision_engine.h
        src/nextfloor/physics/cube_border.h
        src/nextfloor/physics/nearer_collision_engine.h
        src/nextfloor/physics/mesh_physic_factory.h
        src/nextfloor/physics/serial_nearer_collision_engine.h
        src/nextfloor/physics/tbb_nearer_collision_engine.h)

set(polygons_HDRS
        src/nextfloor/polygons/cube.h
        src/nextfloor/polygons/mesh_polygon.h
        src/nextfloor/polygons/mesh_polygon_factory.h)

set(renderer_HDRS
        src/nextfloor/renderer/cube_gl_renderer_engine.h
        src/nextfloor/renderer/fragment_gl_shader.h
        src/nextfloor/renderer/gl_renderer_engine.h
        src/nextfloor/renderer/gl_renderer_factory.h
        src/nextfloor/renderer/gl_scene_input.h
        src/nextfloor/renderer/gl_scene_window.h
        src/nextfloor/renderer/gl_shader.h
        src/nextfloor/renderer/gl_shader_factory.h
        src/nextfloor/renderer/shader.h
        src/nextfloor/renderer/shader_factory.h
        src/nextfloor/renderer/vertex_gl_shader.h)

set(nextfloor_HDRS
        ${core_HDRS}
        ${factory_HDRS}
        ${gameplay_HDRS}
        ${grid_HDRS}
        ${objects_HDRS}
        ${physics_HDRS}
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
add_definitions("-std=gnu++17 -Wall -fno-exceptions")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${Config++_INCLUDE_DIRS})
target_link_libraries(nextfloor ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} config++ tbb)
target_compile_features(nextfloor PUBLIC cxx_nullptr)
