/**
 *  @file object_grid.cc
 *  @brief ObjectGrid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/object_grid.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {


ObjectGrid::ObjectGrid(EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension)
:Grid(owner, boxes_count, box_dimension)
{
    InitBoxes();
}

std::unique_ptr<EngineGridBox> ObjectGrid::AllocateGridBox(glm::ivec3 grid_coords)
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeGridBox(grid_coords, this);
}

ObjectGrid::~ObjectGrid()
{
    //DeleteGrid();
}

} // namespace grid

} // namespace nextfloor
