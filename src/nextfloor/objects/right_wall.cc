/**
 *  @file right_wall.cc
 *  @brief RightWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/right_wall.h"

namespace nextfloor {

namespace objects {

RightWall::RightWall(const glm::vec3& location, const glm::vec3& scale, const MeshFactory& factory)
      : DepthWall(location, scale, factory)
{}

void RightWall::PrepareDraw(const Camera& active_camera)
{
    if (parent_->IsRightPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    DepthWall::PrepareDraw(active_camera);
}

}  // namespace objects

}  // namespace nextfloor
