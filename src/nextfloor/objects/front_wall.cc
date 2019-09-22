/**
 *  @file front_wall.cc
 *  @brief FrontWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/front_wall.h"

namespace nextfloor {

namespace objects {

FrontWall::FrontWall(const glm::vec3& location, const glm::vec3& scale) : WidthWall(location, scale) {}

void FrontWall::PrepareDraw()
{
    if (parent_->IsFrontPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    WidthWall::PrepareDraw();
}

}  // namespace objects

}  // namespace nextfloor
