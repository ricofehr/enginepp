/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_UNIVERSE_H_
#define NEXTFLOOR_PLAYGROUND_UNIVERSE_H_

#include "nextfloor/playground/ground.h"

#include <memory>
#include <vector>

#include "nextfloor/playground/grid.h"
#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace playground {

/**
 *  @class Universe
 *  @brief Define an Universe (with rooms inside), inherits Model abstract class
 */
class Universe : public Ground {

public:
    Universe(std::unique_ptr<Grid> grid,
             std::unique_ptr<nextfloor::mesh::Border> border,
             std::vector<std::unique_ptr<Ground>> rooms);
    ~Universe() final = default;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_UNIVERSE_H_
