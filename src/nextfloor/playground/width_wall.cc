/**
 *  @file width_wall.cc
 *  @brief WidthWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/width_wall.h"

namespace nextfloor {

namespace playground {

WidthWall::WidthWall(std::unique_ptr<nextfloor::mesh::Border> border,
                     std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    border_ = std::move(border);
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    AddBricks(std::move(wall_bricks));
}

void WidthWall::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.x <= location().x - 6.0f && obj_location.y <= location().y + 2.0f) {
            remove_child(objects_[cnt].get());
            return AddDoor();
        }
    }
}

void WidthWall::AddWindow()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.y >= location().y - 3.0f && obj_location.y <= location().y) {
            if (obj_location.x >= location().x - 3.0f && obj_location.x <= location().x + 3.0f) {
                remove_child(objects_[cnt].get());
                return AddWindow();
            }
        }
    }
}

}  // namespace playground

}  // namespace nextfloor
