/**
 *  @file roof.cc
 *  @brief Roof class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/roof.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace playground {

Roof::Roof(std::unique_ptr<nextfloor::mesh::Border> border,
           std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    border_ = std::move(border);
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    AddBricks(std::move(wall_bricks));
}

void Roof::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.x >= location().x - 3.0f && obj_location.x <= location().x + 3.0f) {
            if (obj_location.z >= location().z - 3.0f && obj_location.z <= location().z + 3.0f) {
                remove_child(objects_[cnt].get());
                return AddDoor();
            }
        }
    }
}


void Roof::AddWindow() {}

void Roof::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    if (parent_->IsTopPositionFilled()) {
        AddDoor();
    }

    Wall::PrepareDraw(view_projection_matrix);
}

}  // namespace playground

}  // namespace nextfloor
