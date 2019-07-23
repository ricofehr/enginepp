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
:Grid(owner, glm::ivec3(kWIDTH_BOXES_COUNT, kHEIGHT_BOXES_COUNT, kDEPTH_BOXES_COUNT), glm::vec3(kBOX_WIDTH, kBOX_HEIGHT, kBOX_DEPTH))
{
    InitBoxes();
}

std::unique_ptr<EngineGridBox> RoomGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeRoomGridBox(grid_coords, this);
}

RoomGrid::~RoomGrid()
{
    //DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
