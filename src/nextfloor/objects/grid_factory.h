/**
 *  @file grid_factory.h
 *  @brief Abstract Factory Class for grids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_GRIDFACTORY_H_
#define NEXTFLOOR_OBJECTS_GRIDFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/grid_box.h"

namespace nextfloor {

namespace objects {

/**
 *  @class GridFactory
 *  @brief Abstract Factory Pattern for Grids
 */
class GridFactory {

public:
    virtual ~GridFactory() = default;

    virtual std::unique_ptr<Grid> MakeUniverseGrid(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Grid> MakeRoomGrid(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Grid> MakeGrid(const glm::vec3& location,
                                           const glm::ivec3& boxes_count,
                                           const glm::vec3& box_dimension) const = 0;
    virtual std::unique_ptr<GridBox> MakeRoomGridBox(const glm::vec3& coords, Grid* grid) const = 0;
    virtual std::unique_ptr<GridBox> MakeUniverseGridBox(const glm::vec3& coords, Grid* grid) const = 0;
    virtual std::unique_ptr<GridBox> MakeGridBox(const glm::vec3& coords, Grid* grid) const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_GRIDFACTORY_H_
