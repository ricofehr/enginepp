/**
 *  @file room_places_grid.cc
 *  @brief RoomPlacesGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


RoomGrid::RoomGrid(const glm::vec3& location, std::unique_ptr<nextfloor::objects::GridBox>*** boxes)
      : WiredGrid(location,
                  glm::ivec3(kWIDTH_BOXES_COUNT, kHEIGHT_BOXES_COUNT, kDEPTH_BOXES_COUNT),
                  glm::vec3(kBOX_WIDTH, kBOX_HEIGHT, kBOX_DEPTH),
                  std::move(boxes))
{}

RoomGrid::~RoomGrid() noexcept
{
    DeleteGrid();
}

}  // namespace grid

}  // namespace nextfloor
