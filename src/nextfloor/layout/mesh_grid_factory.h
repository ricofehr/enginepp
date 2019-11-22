/**
 *  @file mesh_grid-factory.h
 *  @brief MeshGridFactory Class for grid objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_MESHGRIDFACTORY_H_
#define NEXTFLOOR_GRID_MESHGRIDFACTORY_H_

#include "nextfloor/playground/grid_factory.h"

namespace nextfloor {

namespace layout {

/**
 *  @class MeshGridFactory
 *  @brief Concret factory for Grid Objects
 */
class MeshGridFactory : public nextfloor::playground::GridFactory {

public:
    std::unique_ptr<nextfloor::playground::Grid> MakeUniverseGrid(const glm::vec3& location) const final;
    std::unique_ptr<nextfloor::playground::Grid> MakeRoomGrid(const glm::vec3& location) const final;

private:
    std::unique_ptr<nextfloor::mesh::GridBox> MakeGridBox(const glm::ivec3& grid_coords) const;
    std::unique_ptr<nextfloor::mesh::GridBox>*** GenerateBoxes(unsigned int grid_width,
                                                               unsigned int grid_height,
                                                               unsigned int grid_depth) const;
};

}  // namespace layout

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_MESHGRIDFACTORY_H_
