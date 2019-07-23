/**
 *  @file right_wall.cc
 *  @brief RightWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/right_wall.h"

namespace nextfloor {

namespace objects {

RightWall::RightWall(glm::vec3 location, glm::vec3 scale)
    :DepthWall(location, scale) {}

void RightWall::PrepareDraw() noexcept
{
    if (parent_->IsRightPositionFilled()) {
        AddDoor();
    } else {
        AddWindow();
    }
}

} // namespace objects

} // namespace nextfloor
