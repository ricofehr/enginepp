/**
 *  @file mesh_grid.h
 *  @brief MeshGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_MESHGRID_H_
#define NEXTFLOOR_GRID_MESHGRID_H_

#include "nextfloor/grid/wired_grid.h"

#include "nextfloor/grid/mesh_grid_factory.h"
#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class MeshGrid
 *  @brief Defines Grid for standard mesh Objects
 */
class MeshGrid : public WiredGrid {

public:
    MeshGrid(const glm::vec3& location,
             const glm::ivec3& boxes_count,
             const glm::vec3& box_dimension,
             const MeshGridFactory& factory);
    ~MeshGrid() noexcept final;

protected:
    std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(const glm::ivec3& grid_coords,
                                                                 const MeshGridFactory& factory) final;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_MESHGRID_H_
