/**
 *  @file floor.cc
 *  @brief Floor class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/depth_wall.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace playground {

DepthWall::DepthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void DepthWall::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.z <= location().z - 6.0f && obj_location.y <= location().y + 2.0f) {
            remove_child(objects_[cnt].get());
            return AddDoor();
        }
    }
}

void DepthWall::AddWindow()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.y >= location().y - 3.0f && obj_location.y <= location().y) {
            if (obj_location.z >= location().z - 3.0f && obj_location.z <= location().z + 3.0f) {
                remove_child(objects_[cnt].get());
                return AddWindow();
            }
        }
    }
}

}  // namespace playground

}  // namespace nextfloor
