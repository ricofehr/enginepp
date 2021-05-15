/**
 *  @file roof.cc
 *  @brief Roof class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/roof.h"

#include <utility>
#include <glm/glm.hpp>

namespace nextfloor {

namespace playground {

Roof::Roof(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void Roof::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.x >= -kDoorDeltaX && obj_location.x <= kDoorDeltaX) {
            if (obj_location.z >= -kDoorDeltaZ && obj_location.z <= kDoorDeltaZ) {
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
