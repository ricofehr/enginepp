/**
 *  @file width_wall.cc
 *  @brief WidthWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/width_wall.h"

namespace nextfloor {

namespace playground {

WidthWall::WidthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    AddBricks(std::move(wall_bricks));
}

void WidthWall::AddDoor()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.x <= location().x - kDoorDeltaX && obj_location.y <= location().y + kDoorDeltaY) {
            remove_child(objects_[cnt].get());
            return AddDoor();
        }
    }
}

void WidthWall::AddWindow()
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.y >= location().y - kWindowDeltaY && obj_location.y <= location().y) {
            if (obj_location.x >= location().x - kWindowDeltaX && obj_location.x <= location().x + kWindowDeltaX) {
                remove_child(objects_[cnt].get());
                return AddWindow();
            }
        }
    }
}

}  // namespace playground

}  // namespace nextfloor
