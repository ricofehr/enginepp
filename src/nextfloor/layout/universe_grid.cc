/**
 *  @file universe_grid.cc
 *  @brief UniverseGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/layout/universe_grid.h"

#include <utility>

namespace nextfloor {

namespace layout {


UniverseGrid::UniverseGrid(const glm::vec3& location, std::unique_ptr<nextfloor::mesh::GridBox>*** boxes)
      : WiredGrid(location,
                  glm::ivec3(kWidthBoxesCount, kHeightBoxesCount, kDepthBoxesCount),
                  glm::vec3(kBoxWidth, kBoxHeight, kBoxDepth),
                  std::move(boxes))
{}

UniverseGrid::~UniverseGrid() noexcept
{
    DeleteGrid();
}

}  // namespace layout

}  // namespace nextfloor
