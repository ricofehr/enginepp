/**
 *  @file universe_grid.cc
 *  @brief UniverseGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/universe_grid.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace grid {


UniverseGrid::UniverseGrid(nextfloor::objects::Mesh* owner)
      : WiredGrid(owner,
                  glm::ivec3(kWIDTH_BOXES_COUNT, kHEIGHT_BOXES_COUNT, kDEPTH_BOXES_COUNT),
                  glm::vec3(kBOX_WIDTH, kBOX_HEIGHT, kBOX_DEPTH))
{
    InitBoxes();
}

std::unique_ptr<nextfloor::objects::GridBox> UniverseGrid::AllocateGridBox(const glm::ivec3& grid_coords)
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    return factory->MakeUniverseGridBox(grid_coords, this);
}

UniverseGrid::~UniverseGrid() noexcept
{
    DeleteGrid();
}

}  // namespace grid

}  // namespace nextfloor
