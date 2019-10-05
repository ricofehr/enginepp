/**
 *  @file left_wall.cc
 *  @brief LeftWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/left_wall.h"

namespace nextfloor {

namespace objects {

LeftWall::LeftWall(const glm::vec3& location, const glm::vec3& scale) : DepthWall(location, scale) {}

void LeftWall::PrepareDraw(const Camera& active_camera)
{
    if (parent_->IsLeftPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    DepthWall::PrepareDraw(active_camera);
}

}  // namespace objects

}  // namespace nextfloor
