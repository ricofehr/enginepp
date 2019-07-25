/**
 *  @file mesh_places_grid.cc
 *  @brief MeshPlacesGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/mesh_places_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


MeshPlacesGrid::MeshPlacesGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension)
: PlacesGrid(owner, boxes_count, box_dimension)
{
    InitBoxes();
}

std::unique_ptr<nextfloor::objects::GridBox> MeshPlacesGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeGridBox(grid_coords, this);
}

MeshPlacesGrid::~MeshPlacesGrid()
{
    DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
