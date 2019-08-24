/**
 *  @file mesh_grid.h
 *  @brief MeshGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_MESHGRID_H_
#define NEXTFLOOR_GRID_MESHGRID_H_

#include "nextfloor/grid/wired_grid.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class MeshGrid
 *  @brief Defines Grid for standard mesh Objects
 */
class MeshGrid : public WiredGrid {

public:
    MeshGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);

    MeshGrid(MeshGrid&&) = default;
    MeshGrid& operator=(MeshGrid&&) = default;
    MeshGrid(const MeshGrid&) = delete;
    MeshGrid& operator=(const MeshGrid&) = delete;

    virtual ~MeshGrid() override;

protected:
    virtual std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_MESHGRID_H_
