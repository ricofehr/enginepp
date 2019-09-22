/**
 *  @file back_wall.cc
 *  @brief BackWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/back_wall.h"

namespace nextfloor {

namespace objects {

BackWall::BackWall(const glm::vec3& location, const glm::vec3& scale) : WidthWall(location, scale) {}

void BackWall::PrepareDraw()
{
    if (parent_->IsBackPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    WidthWall::PrepareDraw();
}

}  // namespace objects

}  // namespace nextfloor
