/**
 *  @file mesh_grid.cc
 *  @brief MeshGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/mesh_grid.h"


#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


MeshGrid::MeshGrid(const glm::vec3& location,
                   const glm::ivec3& boxes_count,
                   const glm::vec3& box_dimension,
                   const MeshGridFactory& factory)
      : WiredGrid(location, boxes_count, box_dimension)
{
    InitBoxes(factory);
}

std::unique_ptr<nextfloor::objects::GridBox> MeshGrid::AllocateGridBox(const glm::ivec3& grid_coords,
                                                                       const MeshGridFactory& factory)
{
    return factory.MakeGridBox(grid_coords, this);
}

MeshGrid::~MeshGrid() noexcept
{
    DeleteGrid();
}

}  // namespace grid

}  // namespace nextfloor
