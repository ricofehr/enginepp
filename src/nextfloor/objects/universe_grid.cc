/**
 *  @file universe_grid.cc
 *  @brief UniverseGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe_grid.h"

#include "nextfloor/objects/room.h"
#include "nextfloor/objects/universe_grid_box.h"

namespace nextfloor {

namespace objects {


UniverseGrid::UniverseGrid(EngineObject* owner)
: Grid(owner)
{
    type_ = kGRID_UNIVERSE;
    InitGrid();
}

std::unique_ptr<EngineGridBox> UniverseGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    return std::make_unique<UniverseGridBox>(grid_coords, this);
}

void UniverseGrid::InitLeftDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.x != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x-1,coords.y,coords.z)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_LEFT);
    } else {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomWindow(Room::kSIDE_LEFT);
    }
}

void UniverseGrid::InitRightDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.x != count_width_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x+1,coords.y,coords.z)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_RIGHT);
    } else {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomWindow(Room::kSIDE_RIGHT);
    }
}


void UniverseGrid::InitFrontDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.z != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y,coords.z-1)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_FRONT);
    } else {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomWindow(Room::kSIDE_FRONT);
    }
}

void UniverseGrid::InitBackDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.z != count_depth_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y,coords.z+1)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_BACK);
    } else {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomWindow(Room::kSIDE_BACK);
    }
}

void UniverseGrid::InitFloorDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.y != 0 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y-1,coords.z)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_FLOOR);
    }
}

void UniverseGrid::InitRoofDoorAndWindowForRoom(glm::ivec3 coords) noexcept
{
    if (coords.y != count_height_boxes() - 1 &&
        IsPositionInTheGridEmpty(glm::ivec3(coords.x,coords.y+1,coords.z)) == kGRID_USED) {
        dynamic_cast<UniverseGridBox*>(getGridBox(coords))->AddRoomDoor(Room::kSIDE_ROOF);
    }
}


void UniverseGrid::InitDoorsAndWindowsForRooms() noexcept
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

    // for (auto &o : objects_) {
    //     auto r = dynamic_cast<Room*>(o.get());
    //     r->addDoor(Room::kSIDE_LEFT);
    //     r->addDoor(Room::kSIDE_FRONT);
    //     r->addDoor(Room::kSIDE_BACK);
    //     r->addDoor(Room::kSIDE_RIGHT);
    //     r->addDoor(Room::kSIDE_FLOOR);
    //     r->addDoor(Room::kSIDE_ROOF);
    // }
    // for (auto &o : objects_) {
    //     /* Objects contained in Universe are Rooms, with Doors and Windows */
    //     auto r = dynamic_cast<Room*>(o.get());

    //     auto i = r->placements()[0][0];
    //     auto j = r->placements()[0][1];
    //     auto k = r->placements()[0][2];

    //     if (i != 0 && IsPositionInTheGridEmpty(i-1,j,k) == kGRID_USED) {
    //         r->addDoor(kLEFT);
    //     }

    //     if (i != grid_x_-1 && IsPositionInTheGridEmpty(i+1,j,k) == kGRID_USED) {
    //         r->addDoor(kRIGHT);
    //     } else {
    //         r->addWindow(kRIGHT);
    //     }

    //     if (j != 0 && IsPositionInTheGridEmpty(i,j-1,k) == kGRID_USED) {
    //         r->addDoor(kFLOOR);
    //     }

    //     if (j != grid_y_-1 && IsPositionInTheGridEmpty(i,j+1,k) == kGRID_USED) {
    //         r->addDoor(kROOF);
    //     }

    //     if (k != 0 && IsPositionInTheGridEmpty(i,j,k-1) == kGRID_USED) {
    //         r->addDoor(kFRONT);
    //     }

    //     if (k != grid_z_-1 && IsPositionInTheGridEmpty(i,j,k+1) == kGRID_USED) {
    //         r->addDoor(kBACK);
    //     }
    //     else if (i != grid_x_-1 && IsPositionInTheGridEmpty(i+1,j,k) == kGRID_USED) {
    //         r->addWindow(kBACK);
    //     }
    //}
}

UniverseGrid::~UniverseGrid()
{
    DeleteGrid();
}

} // namespace objects

} // namespace nextfloor
