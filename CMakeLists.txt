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

set(action_SRCS
        src/nextfloor/action/movedown_action.cc
        src/nextfloor/action/moveleft_action.cc
        src/nextfloor/action/moveright_action.cc
        src/nextfloor/action/moveup_action.cc
        src/nextfloor/action/sprite_action_factory.cc)

set(camera_SRCS
        src/nextfloor/camera/game_camera_factory.cc
        src/nextfloor/camera/head_camera.cc)

set(character_SRCS
        src/nextfloor/character/game_character_factory.cc
        src/nextfloor/character/player.cc)

set(core_SRCS
        src/nextfloor/core/common_services.cc
        src/nextfloor/core/file_config_parser.cc
        src/nextfloor/core/game_file_io.cc
        src/nextfloor/core/program_exit.cc
        src/nextfloor/core/pseudo_random_generator.cc
        src/nextfloor/core/services_core_factory.cc
        src/nextfloor/core/terminal_log.cc)

set(gameplay_SRCS
        src/nextfloor/gameplay/demo_game_factory.cc
        src/nextfloor/gameplay/game_level.cc
        src/nextfloor/gameplay/game_loop.cc
        src/nextfloor/gameplay/game_timer.cc)

set(hid_SRCS
        src/nextfloor/hid/game_input_handler.cc
        src/nextfloor/hid/mouse_hid_factory.cc
        src/nextfloor/hid/mouse_keyboard.cc)

set(layout_SRCS
        src/nextfloor/layout/mesh_grid_factory.cc
        src/nextfloor/layout/room_grid.cc
        src/nextfloor/layout/universe_grid.cc
        src/nextfloor/layout/wired_grid.cc
        src/nextfloor/layout/wired_grid_box.cc)

set(mesh_SRCS
        src/nextfloor/mesh/placement_mesh.cc
        src/nextfloor/mesh/composite_mesh.cc
        src/nextfloor/mesh/drawing_mesh.cc
        src/nextfloor/mesh/dynamic_mesh.cc)

set(physic_SRCS
        src/nextfloor/physic/cl_nearer_collision_engine.cc
        src/nextfloor/physic/cube_border.cc
        src/nextfloor/physic/game_collision_engine_factory.cc
        src/nextfloor/physic/mesh_border_factory.cc
        src/nextfloor/physic/nearer_collision_engine.cc
        src/nextfloor/physic/serial_nearer_collision_engine.cc
        src/nextfloor/physic/tbb_nearer_collision_engine.cc)

set(playground_SRCS
        src/nextfloor/playground/back_wall.cc
        src/nextfloor/playground/depth_wall.cc
        src/nextfloor/playground/floor.cc
        src/nextfloor/playground/front_wall.cc
        src/nextfloor/playground/game_ground_factory.cc
        src/nextfloor/playground/ground.cc
        src/nextfloor/playground/left_wall.cc
        src/nextfloor/playground/right_wall.cc
        src/nextfloor/playground/roof.cc
        src/nextfloor/playground/room.cc
        src/nextfloor/playground/universe.cc
        src/nextfloor/playground/wall.cc
        src/nextfloor/playground/width_wall.cc)

set(polygon_SRCS
        src/nextfloor/polygon/cube.cc
        src/nextfloor/polygon/mesh_polygon.cc
        src/nextfloor/polygon/mesh_polygon_factory.cc)

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

set(thing_SRCS
        src/nextfloor/thing/game_thing_factory.cc
        src/nextfloor/thing/rock.cc
        src/nextfloor/thing/wall_brick.cc)

set(nextfloor_SRCS
        ${action_SRCS}
        ${camera_SRCS}
        ${character_SRCS}
        ${core_SRCS}
        ${gameplay_SRCS}
        ${layout_SRCS}
        ${hid_SRCS}
        ${mesh_SRCS}
        ${physic_SRCS}
        ${playground_SRCS}
        ${polygon_SRCS}
        ${renderer_SRCS}
        ${thing_SRCS}
        src/nextfloor.cc)

set(action_HDRS
        src/nextfloor/action/movedown_action.h
        src/nextfloor/action/moveleft_action.h
        src/nextfloor/action/moveright_action.h
        src/nextfloor/action/moveup_action.h
        src/nextfloor/action/sprite_action_factory.h)

set(camera_HDRS
        src/nextfloor/camera/game_camera_factory.h
        src/nextfloor/camera/head_camera.h)

set(character_HDRS
        src/nextfloor/character/camera.h
        src/nextfloor/character/camera_factory.h
        src/nextfloor/character/character.h
        src/nextfloor/character/character_factory.h
        src/nextfloor/character/game_character_factory.h
        src/nextfloor/character/player.h)

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
        src/nextfloor/gameplay/game_factory.h
        src/nextfloor/gameplay/game_level.h
        src/nextfloor/gameplay/game_loop.h
        src/nextfloor/gameplay/game_timer.h
        src/nextfloor/gameplay/hid.h
        src/nextfloor/gameplay/hid_factory.h
        src/nextfloor/gameplay/input_handler.h
        src/nextfloor/gameplay/level.h
        src/nextfloor/gameplay/loop.h
        src/nextfloor/gameplay/renderer_engine.h
        src/nextfloor/gameplay/renderer_factory.h
        src/nextfloor/gameplay/scene_input.h
        src/nextfloor/gameplay/scene_window.h)

set(hid_HDRS
        src/nextfloor/hid/game_input_handler.h
        src/nextfloor/hid/mouse_hid_factory.h
        src/nextfloor/hid/mouse_keyboard.h)

set(layout_HDRS
        src/nextfloor/layout/mesh_grid_factory.h
        src/nextfloor/layout/room_grid.h
        src/nextfloor/layout/universe_grid.h
        src/nextfloor/layout/wired_grid.h
        src/nextfloor/layout/wired_grid_box.h)

set(mesh_HDRS
        src/nextfloor/mesh/border.h
        src/nextfloor/mesh/border_factory.h
        src/nextfloor/mesh/placement_mesh.h
        src/nextfloor/mesh/composite_mesh.h
        src/nextfloor/mesh/default_mesh.h
        src/nextfloor/mesh/drawing_mesh.h
        src/nextfloor/mesh/dynamic_mesh.h
        src/nextfloor/mesh/grid_box.h
        src/nextfloor/mesh/mesh.h
        src/nextfloor/mesh/polygon.h
        src/nextfloor/mesh/polygon_factory.h)

set(physic_HDRS
        src/nextfloor/physic/cl_nearer_collision_engine.h
        src/nextfloor/physic/collision_engine.h
        src/nextfloor/physic/collision_engine_factory.h
        src/nextfloor/physic/cube_border.h
        src/nextfloor/physic/game_collision_engine_factory.h
        src/nextfloor/physic/mesh_border_factory.h
        src/nextfloor/physic/nearer_collision_engine.h
        src/nextfloor/physic/serial_nearer_collision_engine.h
        src/nextfloor/physic/tbb_nearer_collision_engine.h)

set(playground_HDRS
        src/nextfloor/playground/back_wall.h
        src/nextfloor/playground/depth_wall.h
        src/nextfloor/playground/floor.h
        src/nextfloor/playground/front_wall.h
        src/nextfloor/playground/game_ground_factory.h
        src/nextfloor/playground/grid.h
        src/nextfloor/playground/grid_factory.h
        src/nextfloor/playground/ground.h
        src/nextfloor/playground/ground_factory.h
        src/nextfloor/playground/left_wall.h
        src/nextfloor/playground/right_wall.h
        src/nextfloor/playground/roof.h
        src/nextfloor/playground/room.h
        src/nextfloor/playground/universe.h
        src/nextfloor/playground/wall.h
        src/nextfloor/playground/width_wall.h)

set(polygon_HDRS
        src/nextfloor/polygon/cube.h
        src/nextfloor/polygon/mesh_polygon.h
        src/nextfloor/polygon/mesh_polygon_factory.h)

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

set(thing_HDRS
        src/nextfloor/thing/game_thing_factory.h
        src/nextfloor/thing/rock.h
        src/nextfloor/thing/thing.h
        src/nextfloor/thing/thing_factory.h
        src/nextfloor/thing/wall_brick.h)

set(nextfloor_HDRS
        ${action_HDRS}
        ${camera_HDRS}
        ${character_HDRS}
        ${core_HDRS}
        ${gameplay_HDRS}
        ${layout_HDRS}
        ${hid_HDRS}
        ${mesh_HDRS}
        ${physic_HDRS}
        ${playground_HDRS}
        ${polygon_HDRS}
        ${renderer_HDRS}
        ${thing_HDRS})

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
