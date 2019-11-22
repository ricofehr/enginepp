/**
 *  @file universe_grid.cc
 *  @brief UniverseGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/layout/universe_grid.h"

namespace nextfloor {

namespace layout {


UniverseGrid::UniverseGrid(const glm::vec3& location, std::unique_ptr<nextfloor::mesh::GridBox>*** boxes)
      : WiredGrid(location,
                  glm::ivec3(kWIDTH_BOXES_COUNT, kHEIGHT_BOXES_COUNT, kDEPTH_BOXES_COUNT),
                  glm::vec3(kBOX_WIDTH, kBOX_HEIGHT, kBOX_DEPTH),
                  std::move(boxes))
{}

UniverseGrid::~UniverseGrid() noexcept
{
    DeleteGrid();
}

}  // namespace layout

}  // namespace nextfloor
