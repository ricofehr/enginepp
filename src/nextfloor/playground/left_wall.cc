/**
 *  @file left_wall.cc
 *  @brief LeftWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/left_wall.h"

namespace nextfloor {

namespace playground {

LeftWall::LeftWall(std::unique_ptr<nextfloor::mesh::Border> border,
                   std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
      : DepthWall(std::move(border), std::move(wall_bricks))
{}

void LeftWall::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    if (parent_->IsLeftPositionFilled()) {
        AddDoor();
    }
    else {
        AddWindow();
    }

    DepthWall::PrepareDraw(view_projection_matrix);
}

}  // namespace playground

}  // namespace nextfloor
