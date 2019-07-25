/**
 *  @file mesh_places_grid.h
 *  @brief MeshPlacesGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_MESHPLACESGRID_H_
#define NEXTFLOOR_GRID_MESHPLACESGRID_H_

#include "nextfloor/grid/places_grid.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class MeshPlacesGrid
 *  @brief Defines Grid for standard mesh Objects
 */
class MeshPlacesGrid : public PlacesGrid {

public:

    MeshPlacesGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);
    MeshPlacesGrid(MeshPlacesGrid&&) = default;
    MeshPlacesGrid& operator=(MeshPlacesGrid&&) = default;
    MeshPlacesGrid(const MeshPlacesGrid&) = delete;
    MeshPlacesGrid& operator=(const MeshPlacesGrid&) = delete;
    virtual ~MeshPlacesGrid() override;

protected:

    virtual std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_MESHPLACESGRID_H_
