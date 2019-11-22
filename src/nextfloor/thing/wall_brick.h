/**
 *  @file wall_brick.h
 *  @brief WallBrick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_THING_WALLBRICK_H_
#define NEXTFLOOR_THING_WALLBRICK_H_

#include "nextfloor/thing/thing.h"

#include <memory>

namespace nextfloor {

namespace thing {

/**
 *  @class WallBrick
 *  @brief WallBrick model, inherits Model abstract class
 */
class WallBrick : public Thing {

public:
    WallBrick(std::unique_ptr<nextfloor::mesh::Border> border,
              std::vector<std::unique_ptr<nextfloor::mesh::Polygon>> bricks);
    ~WallBrick() final = default;
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_THING_WALLBRICK_H_