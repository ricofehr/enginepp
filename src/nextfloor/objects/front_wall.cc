/**
 *  @file front_wall.cc
 *  @brief FrontWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/front_wall.h"

namespace nextfloor {

namespace objects {

FrontWall::FrontWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks)
      : WidthWall(std::move(border), std::move(wall_bricks))
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
