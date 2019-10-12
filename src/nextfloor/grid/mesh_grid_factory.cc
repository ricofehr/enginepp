/**
 *  @file mesh_grid_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/grid/mesh_grid_factory.h"

#include "nextfloor/grid/mesh_grid.h"
#include "nextfloor/grid/room_grid.h"
#include "nextfloor/grid/universe_grid.h"
#include "nextfloor/grid/room_grid_box.h"
#include "nextfloor/grid/universe_grid_box.h"
#include "nextfloor/grid/wired_grid_box.h"


namespace nextfloor {

namespace grid {

std::unique_ptr<nextfloor::objects::Grid> MeshGridFactory::MakeUniverseGrid(const glm::vec3& location) const
{
    return std::make_unique<UniverseGrid>(location, *this);
}

std::unique_ptr<nextfloor::objects::Grid> MeshGridFactory::MakeRoomGrid(const glm::vec3& location) const
{
    return std::make_unique<RoomGrid>(location, *this);
}

std::unique_ptr<nextfloor::objects::Grid> MeshGridFactory::MakeGrid(const glm::vec3& location,
                                                                    const glm::ivec3& boxes_count,
                                                                    const glm::vec3& box_dimension) const
{
    return std::make_unique<MeshGrid>(location, boxes_count, box_dimension, *this);
}

std::unique_ptr<nextfloor::objects::GridBox> MeshGridFactory::MakeRoomGridBox(const glm::vec3& coords,
                                                                              nextfloor::objects::Grid* room_grid) const
{
    return std::make_unique<RoomGridBox>(coords, room_grid);
}

std::unique_ptr<nextfloor::objects::GridBox>
  MeshGridFactory::MakeUniverseGridBox(const glm::vec3& coords, nextfloor::objects::Grid* universe_grid) const
{
    return std::make_unique<UniverseGridBox>(coords, universe_grid);
}

std::unique_ptr<nextfloor::objects::GridBox> MeshGridFactory::MakeGridBox(const glm::vec3& grid_coords,
                                                                          nextfloor::objects::Grid* grid) const
{
    return std::make_unique<WiredGridBox>(grid_coords, grid);
}

}  // namespace grid

}  // namespace nextfloor