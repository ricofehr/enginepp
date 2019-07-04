/**
 *  @file room_grid.cc
 *  @brief RoomGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_grid.h"

#include "nextfloor/grid/room_grid_box.h"

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
    return std::make_unique<RoomGridBox>(grid_coords, this);
}

RoomGrid::~RoomGrid()
{
    DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
