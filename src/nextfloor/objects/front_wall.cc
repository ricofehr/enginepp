/**
 *  @file front_wall.cc
 *  @brief FrontWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/front_wall.h"

namespace nextfloor {

namespace objects {

FrontWall::FrontWall(glm::vec3 location, glm::vec3 scale) : WidthWall(location, scale) {}

void FrontWall::PrepareDraw() noexcept
{
    if (parent_->IsFrontPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }
}

}  // namespace objects

}  // namespace nextfloor
