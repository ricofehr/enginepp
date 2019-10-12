/**
 *  @file back_wall.cc
 *  @brief BackWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/back_wall.h"

namespace nextfloor {

namespace objects {

BackWall::BackWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks)
      : WidthWall(std::move(border), std::move(wall_bricks))
{}

void BackWall::PrepareDraw(const Camera& active_camera)
{
    if (parent_->IsBackPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    WidthWall::PrepareDraw(active_camera);
}

}  // namespace objects

}  // namespace nextfloor
