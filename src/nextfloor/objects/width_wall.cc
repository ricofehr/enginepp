/**
 *  @file width_wall.cc
 *  @brief WidthWall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/width_wall.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

WidthWall::WidthWall(const glm::vec3& location, const glm::vec3& scale)
{
    using nextfloor::core::CommonServices;
    border_ = CommonServices::getFactory().MakeBorder(location, glm::vec3(scale));
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    AddBricks(location - scale, location + scale);
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

}  // namespace objects

}  // namespace nextfloor
