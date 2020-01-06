/**
 *  @file room_places_grid.cc
 *  @brief RoomPlacesGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/layout/room_grid.h"

#include <utility>

namespace nextfloor {

namespace layout {


RoomGrid::RoomGrid(const glm::vec3& location, std::unique_ptr<nextfloor::mesh::GridBox>*** boxes)
      : WiredGrid(location,
                  glm::ivec3(kWidthBoxesCount, kHeightBoxesCount, kDepthBoxesCount),
                  glm::vec3(kBoxWidth, kBoxHeight, kBoxDepth),
                  std::move(boxes))
{}

RoomGrid::~RoomGrid() noexcept
{
    DeleteGrid();
}

}  // namespace layout

}  // namespace nextfloor
