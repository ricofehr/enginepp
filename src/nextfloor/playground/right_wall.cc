/**
 *  @file right_wall.cc
 *  @brief RightWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/right_wall.h"

namespace nextfloor {

namespace playground {

RightWall::RightWall(std::unique_ptr<nextfloor::mesh::Border> border,
                     std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
      : DepthWall(std::move(border), std::move(wall_bricks))
{}

void RightWall::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    if (parent_->IsRightPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    DepthWall::PrepareDraw(view_projection_matrix);
}

}  // namespace playground

}  // namespace nextfloor
