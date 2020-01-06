/**
 *  @file floor.cc
 *  @brief Floor class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/floor.h"

#include <utility>
#include <glm/glm.hpp>

namespace nextfloor {

namespace playground {

Floor::Floor(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void Floor::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.x >= location().x - kDoorDeltaX && obj_location.x <= location().x + kDoorDeltaX) {
            if (obj_location.z >= location().z - kDoorDeltaZ && obj_location.z <= location().z + kDoorDeltaZ) {
                remove_child(objects_[cnt].get());
                return AddDoor();
            }
        }
    }
}

void Floor::AddWindow() {}

void Floor::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    if (parent_->IsBottomPositionFilled()) {
        AddDoor();
    }

    Wall::PrepareDraw(view_projection_matrix);
}

}  // namespace playground

}  // namespace nextfloor
