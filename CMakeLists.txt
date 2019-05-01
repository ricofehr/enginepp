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

set(graphics_SRCS
    src/nextfloor/graphics/shape3d.cc
    src/nextfloor/graphics/quad.cc
    src/nextfloor/graphics/cube.cc
    src/nextfloor/graphics/border.cc)

set(core_SRCS
    src/nextfloor/core/config_engine.cc
    src/nextfloor/core/global_timer.cc
    src/nextfloor/core/file_io.cc)

set(hid_SRCS
    src/nextfloor/hid/input_handler.cc
    src/nextfloor/hid/keyboard_mouse.cc
    src/nextfloor/hid/pad.cc)

set(job_SRCS
    src/nextfloor/job/game_loop.cc)

set(physics_SRCS
    src/nextfloor/physics/collision_engine.cc
    src/nextfloor/physics/cl_collision_engine.cc
    src/nextfloor/physics/tbb_collision_engine.cc
    src/nextfloor/physics/serial_collision_engine.cc)

set(renderer_SRCS
    src/nextfloor/renderer/shader.cc
    src/nextfloor/renderer/fragment_shader.cc
    src/nextfloor/renderer/vertex_shader.cc)

set(universe_SRCS
    src/nextfloor/universe/commands/fire_command.cc
    src/nextfloor/universe/commands/jump_command.cc
    src/nextfloor/universe/commands/movedown_command.cc
    src/nextfloor/universe/commands/moveleft_command.cc
    src/nextfloor/universe/commands/moveright_command.cc
    src/nextfloor/universe/commands/moveup_command.cc
    src/nextfloor/universe/commands/run_command.cc
    src/nextfloor/universe/dynamic/camera.cc
    src/nextfloor/universe/dynamic/brick.cc
    src/nextfloor/universe/factory/random_universe_factory.cc
    src/nextfloor/universe/stationary/wall.cc
    src/nextfloor/universe/model3d.cc
    src/nextfloor/universe/room.cc
    src/nextfloor/universe/universe.cc)

set(nextfloor_SRCS
    ${graphics_SRCS}
    ${core_SRCS}
    ${hid_SRCS}
    ${job_SRCS}
    ${renderer_SRCS}
    ${physics_SRCS}
    ${universe_SRCS}
    src/nextfloor.cc)

set(core_HDRS
    src/nextfloor/core/config_engine.h
    src/nextfloor/core/global_timer.h
    src/nextfloor/core/file_io.h)

set(graphics_HDRS
    src/nextfloor/graphics/shape3d.h
    src/nextfloor/graphics/quad.h
    src/nextfloor/graphics/cube.h
    src/nextfloor/graphics/border.h)

set(hid_HDRS
    src/nextfloor/hid/hid.h
    src/nextfloor/hid/input_handler.h
    src/nextfloor/hid/keyboard_mouse.h
    src/nextfloor/hid/pad.h)

set(job_HDRS
    src/nextfloor/job/game_loop.h)

set(physics_HDRS
    src/nextfloor/physics/collision_engine.h
    src/nextfloor/physics/cl_collision_engine.h
    src/nextfloor/physics/tbb_collision_engine.h
    src/nextfloor/physics/serial_collision_engine.h)

set(renderer_HDRS
    src/nextfloor/renderer/shader.h
    src/nextfloor/renderer/fragment_shader.h
    src/nextfloor/renderer/vertex_shader.h)

set(universe_HDRS
    src/nextfloor/universe/commands/command.h
    src/nextfloor/universe/commands/fire_command.h
    src/nextfloor/universe/commands/jump_command.h
    src/nextfloor/universe/commands/movedown_command.h
    src/nextfloor/universe/commands/moveleft_command.h
    src/nextfloor/universe/commands/moveright_command.h
    src/nextfloor/universe/commands/moveup_command.h
    src/nextfloor/universe/commands/run_command.h
    src/nextfloor/universe/factory/random_universe_factory.h
    src/nextfloor/universe/factory/universe_factory.h
    src/nextfloor/universe/dynamic/camera.h
    src/nextfloor/universe/dynamic/brick.h
    src/nextfloor/universe/stationary/wall.h
    src/nextfloor/universe/model3d.h
    src/nextfloor/universe/room.h
    src/nextfloor/universe/universe.h)

set(nextfloor_HDRS
    ${graphics_HDRS}
    ${core_HDRS}
    ${hid_HDRS}
    ${job_HDRS}
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
find_library(SOIL REQUIRED)
add_definitions("-std=gnu++17")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${SOIL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${Config++_INCLUDE_DIRS})
target_link_libraries(nextfloor ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} SOIL glfw ${OpenCL_LIBRARY} config++ tbb)
target_compile_features(nextfloor PUBLIC cxx_nullptr)
