/**
 *  @file wall_brick.cc
 *  @brief WallBrick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall_brick.h"

namespace nextfloor {

namespace objects {

WallBrick::WallBrick(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Polygon>> bricks)
{
    polygons_ = std::move(bricks);
    border_ = std::move(border);
}

}  // namespace objects

}  // namespace nextfloor
