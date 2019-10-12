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

private:
    std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(const glm::ivec3& grid_coords) const;
    std::unique_ptr<nextfloor::objects::GridBox>*** GenerateBoxes(unsigned int grid_width,
                                                                  unsigned int grid_height,
                                                                  unsigned int grid_depth) const;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_MESHGRIDFACTORY_H_
