/**
 *  @file right_wall.cc
 *  @brief RightWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/right_wall.h"

namespace nextfloor {

namespace objects {

RightWall::RightWall(const glm::vec3& location, const glm::vec3& scale) : DepthWall(location, scale) {}

void RightWall::PrepareDraw()
{
    if (parent_->IsRightPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    DepthWall::PrepareDraw();
}

}  // namespace objects

}  // namespace nextfloor
