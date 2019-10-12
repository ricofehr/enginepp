/**
 *  @file right_wall.cc
 *  @brief RightWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/right_wall.h"

namespace nextfloor {

namespace objects {

RightWall::RightWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks)
      : DepthWall(std::move(border), std::move(wall_bricks))
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
