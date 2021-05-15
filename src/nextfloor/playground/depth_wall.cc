/**
 *  @file floor.cc
 *  @brief Floor class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/depth_wall.h"

#include <utility>
#include <glm/glm.hpp>

namespace nextfloor {

namespace playground {

DepthWall::DepthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void DepthWall::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.z <= -kDoorDeltaZ && obj_location.y <= kDoorDeltaY) {
            remove_child(objects_[cnt].get());
            return AddDoor();
        }
    }
}

void DepthWall::AddWindow()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        glm::vec3 obj_location = objects_[cnt]->location();
        if (obj_location.y >= -kWindowDeltaY && obj_location.y <= 0.0f) {
            if (obj_location.z >= -kWindowDeltaZ && obj_location.z <= kWindowDeltaZ) {
                remove_child(objects_[cnt].get());
                return AddWindow();
            }
        }
    }
}

}  // namespace playground

}  // namespace nextfloor
