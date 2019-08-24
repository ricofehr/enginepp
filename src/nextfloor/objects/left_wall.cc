/**
 *  @file left_wall.cc
 *  @brief LeftWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/left_wall.h"

namespace nextfloor {

namespace objects {

LeftWall::LeftWall(glm::vec3 location, glm::vec3 scale)
      : DepthWall(location, scale)
{}

void LeftWall::PrepareDraw() noexcept
{
    if (parent_->IsLeftPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }
}

}  // namespace objects

}  // namespace nextfloor
