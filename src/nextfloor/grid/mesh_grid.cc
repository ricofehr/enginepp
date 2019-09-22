/**
 *  @file mesh_grid.cc
 *  @brief MeshGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/mesh_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


MeshGrid::MeshGrid(nextfloor::objects::Mesh* owner, const glm::ivec3& boxes_count, const glm::vec3& box_dimension)
      : WiredGrid(owner, boxes_count, box_dimension)
{
    InitBoxes();
}

std::unique_ptr<nextfloor::objects::GridBox> MeshGrid::AllocateGridBox(const glm::ivec3& grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory().MakeGridBox(grid_coords, this);
}

MeshGrid::~MeshGrid() noexcept
{
    DeleteGrid();
}

}  // namespace grid

}  // namespace nextfloor
