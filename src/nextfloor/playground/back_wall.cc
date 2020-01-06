/**
 *  @file back_wall.cc
 *  @brief BackWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/back_wall.h"

#include <utility>

namespace nextfloor {

namespace playground {

BackWall::BackWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
      : WidthWall(std::move(wall_bricks))
{}

void BackWall::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    if (parent_->IsBackPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    WidthWall::PrepareDraw(view_projection_matrix);
}

}  // namespace playground

}  // namespace nextfloor
