/**
 *  @file front_wall.cc
 *  @brief FrontWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/front_wall.h"

namespace nextfloor {

namespace objects {

FrontWall::FrontWall(const glm::vec3& location, const glm::vec3& scale, const MeshFactory& factory)
      : WidthWall(location, scale, factory)
{}

void FrontWall::PrepareDraw(const Camera& active_camera)
{
    if (parent_->IsFrontPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    WidthWall::PrepareDraw(active_camera);
}

}  // namespace objects

}  // namespace nextfloor
