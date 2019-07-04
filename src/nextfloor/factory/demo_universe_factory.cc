/**
 *  @file demo_universe_factory.cc
 *  @brief DemoUniverseFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/factory/demo_universe_factory.h"

#include <string>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"
#include "nextfloor/renderer/gl_cube_renderer.h"

namespace nextfloor {

namespace factory {

namespace {

glm::vec3 CalculateScaleWallXFixed(nextfloor::objects::EngineGrid* grid)
{
    auto box_counts = grid->box_counts();
    auto box_dimension = grid->box_dimension();
    return glm::vec3(box_dimension.x/4, (box_counts.y/6)*box_dimension.y, (box_counts.z/8)*box_dimension.z);
}

glm::vec3 CalculateScaleWallYFixed(nextfloor::objects::EngineGrid* grid)
{
    auto box_counts = grid->box_counts();
    auto box_dimension = grid->box_dimension();
    return glm::vec3((box_counts.x/8)*box_dimension.x, box_dimension.y/4, (box_counts.z/8)*box_dimension.z);
}

glm::vec3 CalculateScaleWallZFixed(nextfloor::objects::EngineGrid* grid)
{
    auto box_counts = grid->box_counts();
    auto box_dimension = grid->box_dimension();
    return glm::vec3((box_counts.x/8)*box_dimension.x, (box_counts.y/6)*box_dimension.y, box_dimension.z/4);
}

glm::vec4 CalculateLocationWallXFixed(nextfloor::objects::EngineGrid* grid, glm::ivec3 grid_coords)
{
    auto grid0 = grid->CalculateFirstPointInGrid();
    auto box_dimension = grid->box_dimension();
    auto scale_wall = CalculateScaleWallXFixed(grid);

    auto location_wall = glm::vec4(grid0, 0.0f) + glm::vec4(0.0f, scale_wall[1], scale_wall[2], 0.0f);
    location_wall += glm::vec4(grid_coords.x*box_dimension.x, grid_coords.y*box_dimension.y, grid_coords.z*box_dimension.z, 0.0f);

    /* padding */
    if (grid_coords.x == 0) {
        location_wall.x += box_dimension.x/4;
    } else {
        location_wall.x -= box_dimension.x/4;
    }

    return location_wall;
}

glm::vec4 CalculateLocationWallYFixed(nextfloor::objects::EngineGrid* grid, glm::ivec3 grid_coords)
{
    auto grid0 = grid->CalculateFirstPointInGrid();
    auto box_dimension = grid->box_dimension();
    auto scale_wall = CalculateScaleWallYFixed(grid);

    auto location_wall = glm::vec4(grid0, 0.0f) + glm::vec4(scale_wall[0], 0.0f, scale_wall[2], 0.0f);
    location_wall += glm::vec4(grid_coords.x*box_dimension.x, grid_coords.y*box_dimension.y, grid_coords.z*box_dimension.z, 0.0f);

    /* padding */
    if (grid_coords.y == 0) {
        location_wall.y += box_dimension.y/4;
    } else {
        location_wall.y -= box_dimension.y/4;
    }

    return location_wall;
}

glm::vec4 CalculateLocationWallZFixed(nextfloor::objects::EngineGrid* grid, glm::ivec3 grid_coords)
{
    auto grid0 = grid->CalculateFirstPointInGrid();
    auto box_dimension = grid->box_dimension();
    auto scale_wall = CalculateScaleWallZFixed(grid);

    auto location_wall = glm::vec4(grid0, 0.0f) + glm::vec4(scale_wall[0], scale_wall[1], 0.0f, 0.0f);
    location_wall += glm::vec4(grid_coords.x*box_dimension.x, grid_coords.y*box_dimension.y, grid_coords.z*box_dimension.z, 0.0f);

    /* padding */
    if (grid_coords.z == 0) {
        location_wall.z += box_dimension.z/4;
    } else {
        location_wall.z -= box_dimension.z/4;
    }

    return location_wall;
}

}

std::unique_ptr<nextfloor::objects::Universe> DemoUniverseFactory::GenerateUniverse() const
{
    using nextfloor::objects::Room;
    using nextfloor::core::CommonServices;

    auto universe = std::make_unique<nextfloor::objects::Universe>();
    std::vector<Room*> rooms;
    rooms.push_back(GenerateRoom(universe.get()));
    universe->InitDoorsForRooms();

    for (auto &room : rooms) {
        GenerateWalls(room);
        GenerateBrick(room);
        room->grid()->DisplayGrid();
    }

    universe->grid()->DisplayGrid();
    return universe;
}

nextfloor::objects::Room* DemoUniverseFactory::GenerateRoom(nextfloor::objects::Universe* universe) const
{
    using nextfloor::objects::Room;

    auto location = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    auto room_ptr{std::make_unique<Room>(location)};
    room_ptr->add_child(GenerateCamera(location));

    auto room = dynamic_cast<Room*>(universe->add_child(std::move(room_ptr)));
    return room;
}

void DemoUniverseFactory::GenerateWalls(nextfloor::objects::Room* room) const
{
    using nextfloor::objects::Room;

    GenerateWallYFixed(room, Room::kSIDE_FLOOR);
    GenerateWallYFixed(room, Room::kSIDE_ROOF);
    GenerateWallXFixed(room, Room::kSIDE_LEFT);
    GenerateWallXFixed(room, Room::kSIDE_RIGHT);
    GenerateWallZFixed(room, Room::kSIDE_FRONT);
    GenerateWallZFixed(room, Room::kSIDE_BACK);
}

void DemoUniverseFactory::GenerateWallYFixed(nextfloor::objects::Room* room, int side) const
{
    using nextfloor::objects::Wall;
    using nextfloor::objects::Room;
    using nextfloor::renderer::GlCubeRenderer;

    auto grid = room->grid();
    auto box_counts = grid->box_counts();
    auto scale_wall = CalculateScaleWallYFixed(grid);

    tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto i = (side == Room::kSIDE_ROOF) ? box_counts.y : 0;
            auto grid_coords = glm::ivec3(j,i,k);
            if (room->IsInsideWall(grid_coords)) {
                auto location_wall = CalculateLocationWallYFixed(grid, grid_coords);
                auto renderer = std::make_unique<GlCubeRenderer>((i == 0) ? "assets/floor.png" : "assets/sky.png");
                auto wall_ptr{std::make_unique<Wall>(scale_wall, location_wall, std::move(renderer))};
                room->add_child(std::move(wall_ptr));
            }
        });
    });
}

void DemoUniverseFactory::GenerateWallXFixed(nextfloor::objects::Room* room, int side) const
{
    using nextfloor::objects::Wall;
    using nextfloor::objects::Room;
    using nextfloor::renderer::GlCubeRenderer;

    auto grid = room->grid();
    auto box_counts = grid->box_counts();
    auto scale_wall = CalculateScaleWallXFixed(grid);

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto i = (side == Room::kSIDE_RIGHT) ? box_counts.x : 0;
            auto grid_coords = glm::ivec3(i,j,k);
            if (room->IsInsideWall(grid_coords)) {
                auto location_wall = CalculateLocationWallXFixed(grid, grid_coords);
                auto renderer = std::make_unique<GlCubeRenderer>("assets/wall.png");
                auto wall_ptr{std::make_unique<Wall>(scale_wall, location_wall, std::move(renderer))};
                room->add_child(std::move(wall_ptr));
            }
        });
    });
}

void DemoUniverseFactory::GenerateWallZFixed(nextfloor::objects::Room* room, int side) const
{
    using nextfloor::objects::Wall;
    using nextfloor::objects::Room;
    using nextfloor::renderer::GlCubeRenderer;

    auto grid = room->grid();
    auto box_counts = grid->box_counts();
    auto scale_wall = CalculateScaleWallZFixed(grid);

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int k) {
            auto i = (side == Room::kSIDE_BACK) ? box_counts.z : 0;
            auto grid_coords = glm::ivec3(k,j,i);
            if (room->IsInsideWall(grid_coords)) {
                auto location_wall = CalculateLocationWallZFixed(grid, grid_coords);
                auto renderer = std::make_unique<GlCubeRenderer>("assets/wall.png");
                auto wall_ptr{std::make_unique<Wall>(scale_wall, location_wall, std::move(renderer))};
                room->add_child(std::move(wall_ptr));
            }
        });
    });
}

void DemoUniverseFactory::GenerateBrick(nextfloor::objects::Room* room) const
{
    float scale = 1.0f;
    auto grid = room->grid();
    auto location_brick = grid->CalculateFirstPointInGrid() + glm::vec3(3.0f);

    using nextfloor::renderer::GlCubeRenderer;
    auto renderer = std::make_unique<GlCubeRenderer>("assets/brique.png");

    using nextfloor::objects::Brick;
    auto obj = std::make_unique<Brick>(scale,
                                       glm::vec4(location_brick, 0.0f),
                                       std::move(renderer));

    room->add_child(std::move(obj));
}

std::unique_ptr<nextfloor::objects::Camera> DemoUniverseFactory::GenerateCamera(glm::vec3 location) const
{
    using nextfloor::objects::Camera;
    return std::make_unique<Camera>(glm::vec4(location.x, location.y, location.z + 5.0f, 0.0f),
                                    3.14f, 0.0f);
}

} // namespace factory

} // namespace nextfloor
