/**
 *  @file universe_grid.cc
 *  @brief UniverseGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/universe_grid.h"

#include "nextfloor/objects/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


UniverseGrid::UniverseGrid(EngineObject* owner)
   :Grid(owner)
{
    type_ = kGRID_UNIVERSE;
    InitGrid();
}

std::unique_ptr<EngineGridBox> UniverseGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeUniverseGridBox(grid_coords, this);
}

void UniverseGrid::InitLeftDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.x != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x-1,coords.y,coords.z)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_LEFT);
    } else {
        getGridBox(coords)->AddRoomWindow(Room::kSIDE_LEFT);
    }
}

void UniverseGrid::InitRightDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.x != count_width_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x+1,coords.y,coords.z)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_RIGHT);
    } else {
        getGridBox(coords)->AddRoomWindow(Room::kSIDE_RIGHT);
    }
}


void UniverseGrid::InitFrontDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.z != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y,coords.z-1)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_FRONT);
    } else {
        getGridBox(coords)->AddRoomWindow(Room::kSIDE_FRONT);
    }
}

void UniverseGrid::InitBackDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.z != count_depth_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y,coords.z+1)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_BACK);
    } else {
        getGridBox(coords)->AddRoomWindow(Room::kSIDE_BACK);
    }
}

void UniverseGrid::InitFloorDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.y != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y-1,coords.z)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_FLOOR);
    }
}

void UniverseGrid::InitRoofDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    using nextfloor::objects::Room;

    if (coords.y != count_height_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y+1,coords.z)) == kGRID_USED) {
        getGridBox(coords)->AddRoomDoor(Room::kSIDE_ROOF);
    }
}


void UniverseGrid::InitDoorsAndWindows() noexcept
{
    for (auto i = 0; i < count_width_boxes(); i++) {
        for (auto j = 0; j < count_height_boxes(); j++) {
            for (auto k = 0; k < count_depth_boxes(); k++) {
                auto grid_coords = glm::ivec3(i,j,k);

                InitLeftDoorAndWindowForRoom(grid_coords);
                InitRightDoorAndWindowForRoom(grid_coords);
                InitFrontDoorAndWindowForRoom(grid_coords);
                InitBackDoorAndWindowForRoom(grid_coords);
                InitFloorDoorAndWindowForRoom(grid_coords);
                InitRoofDoorAndWindowForRoom(grid_coords);
            }
        }
    }
}

UniverseGrid::~UniverseGrid()
{
    DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
