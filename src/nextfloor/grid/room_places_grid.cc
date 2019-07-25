/**
 *  @file room_places_grid.cc
 *  @brief RoomPlacesGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_places_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


RoomPlacesGrid::RoomPlacesGrid(nextfloor::objects::Mesh* owner)
    : PlacesGrid(owner, glm::ivec3(kWIDTH_BOXES_COUNT, kHEIGHT_BOXES_COUNT, kDEPTH_BOXES_COUNT), glm::vec3(kBOX_WIDTH, kBOX_HEIGHT, kBOX_DEPTH))
{
    InitBoxes();
}

std::unique_ptr<nextfloor::objects::GridBox> RoomPlacesGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeRoomGridBox(grid_coords, this);
}

RoomPlacesGrid::~RoomPlacesGrid()
{
    DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
