/**
 *  @file mesh_grid-factory.h
 *  @brief MeshGridFactory Class for grid objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_MESHGRIDFACTORY_H_
#define NEXTFLOOR_GRID_MESHGRIDFACTORY_H_

#include "nextfloor/objects/grid_factory.h"

namespace nextfloor {

namespace grid {

/**
 *  @class MeshGridFactory
 *  @brief Concret factory for Grid Objects
 */
class MeshGridFactory : public nextfloor::objects::GridFactory {

public:
    std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(const glm::vec3& location) const final;
    std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(const glm::vec3& location) const final;
    std::unique_ptr<nextfloor::objects::Grid> MakeGrid(const glm::vec3& location,
                                                       const glm::ivec3& boxes_count,
                                                       const glm::vec3& box_dimension) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(const glm::vec3& grid_coords,
                                                                 nextfloor::objects::Grid* room_grid) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(const glm::vec3& coords,
                                                                     nextfloor::objects::Grid* grid) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(const glm::vec3& grid_coords,
                                                             nextfloor::objects::Grid* grid) const final;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_MESHGRIDFACTORY_H_
