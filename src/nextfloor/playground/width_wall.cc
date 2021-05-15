/**
 *  @file width_wall.cc
 *  @brief WidthWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/width_wall.h"

#include <utility>
#include <glm/glm.hpp>

namespace nextfloor {

namespace playground {

WidthWall::WidthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void WidthWall::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.x <= -kDoorDeltaX && obj_location.y <= kDoorDeltaY) {
            remove_child(objects_[cnt].get());
            return AddDoor();
        }
    }
}

void WidthWall::AddWindow()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.y >= -kWindowDeltaY && obj_location.y <= 0.0f) {
            if (obj_location.x >= 0.0f - kWindowDeltaX && obj_location.x <= kWindowDeltaX) {
                remove_child(objects_[cnt].get());
                return AddWindow();
            }
        }
    }
}

}  // namespace playground

}  // namespace nextfloor
