/**
 *  @file grid_factory.h
 *  @brief Abstract Factory Class for grids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_GRIDFACTORY_H_
#define NEXTFLOOR_PLAYGROUND_GRIDFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/playground/grid.h"

namespace nextfloor {

namespace playground {

/**
 *  @class GridFactory
 *  @brief Abstract Factory Pattern for Grids
 */
class GridFactory {

public:
    virtual ~GridFactory() = default;

    virtual std::unique_ptr<Grid> MakeUniverseGrid(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Grid> MakeRoomGrid(const glm::vec3& location) const = 0;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_GRIDFACTORY_H_
