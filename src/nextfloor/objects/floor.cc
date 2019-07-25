/**
 *  @file floor.cc
 *  @brief Floor class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/floor.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Floor::Floor(glm::vec3 location, glm::vec3 scale)
{
    using nextfloor::core::CommonServices;
    border_ = CommonServices::getFactory()->MakeBorder(location, scale);
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    grid_ = CommonServices::getFactory()->MakeGrid(this, bricks_count_, brick_dimension_);
    AddBricks(location - scale, location + scale);
}

void Floor::AddDoor() noexcept
{
    for (auto cnt = 0; cnt < objects_.size(); cnt++)
    {
        auto obj_location = objects_[cnt]->location();
        if (obj_location.x >= location().x - 3.0f && obj_location.x <= location().x + 3.0f) {
            if (obj_location.z >= location().z - 3.0f && obj_location.z <= location().z + 3.0f) {
                remove_child(objects_[cnt].get());
                return AddDoor();
            }
        }
    }
}

void Floor::AddWindow() noexcept {}

void Floor::PrepareDraw() noexcept
{
    if (parent_->IsBottomPositionFilled()) {
        AddDoor();
    }
}

} // namespace objects

} // namespace nextfloor
