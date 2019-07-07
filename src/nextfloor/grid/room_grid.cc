/**
 *  @file room_grid.cc
 *  @brief RoomGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


RoomGrid::RoomGrid(EngineObject* owner)
   :Grid(owner)
{
    type_ = kGRID_ROOM;
    InitGrid();
}

std::unique_ptr<EngineGridBox> RoomGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeRoomGridBox(grid_coords, this);
}

RoomGrid::~RoomGrid()
{
    DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
