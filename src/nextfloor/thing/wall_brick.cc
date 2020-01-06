/**
 *  @file wall_brick.cc
 *  @brief WallBrick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/thing/wall_brick.h"

#include <utility>

namespace nextfloor {

namespace thing {

WallBrick::WallBrick(std::unique_ptr<nextfloor::mesh::Border> border,
                     std::vector<std::unique_ptr<nextfloor::mesh::Polygon>> bricks)
{
    polygons_ = std::move(bricks);
    border_ = std::move(border);
}

}  // namespace thing

}  // namespace nextfloor
