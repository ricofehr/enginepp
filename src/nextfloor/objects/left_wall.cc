/**
 *  @file left_wall.cc
 *  @brief LeftWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/left_wall.h"

namespace nextfloor {

namespace objects {

LeftWall::LeftWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks)
      : DepthWall(std::move(border), std::move(wall_bricks))
{}

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
