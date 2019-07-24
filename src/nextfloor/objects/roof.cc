/**
 *  @file roof.cc
 *  @brief Roof class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/roof.h"

#include <memory>
#include <iostream>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Roof::Roof(glm::vec3 location, glm::vec3 scale)
{
    using nextfloor::core::CommonServices;
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(scale));
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    grid_ = CommonServices::getFactory()->MakeGrid(this, bricks_count_, brick_dimension_);
    AddBricks(location - scale, location + scale);
    grid_->DisplayGrid();
}

void Roof::AddDoor() noexcept
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


void Roof::AddWindow() noexcept {}

void Roof::PrepareDraw() noexcept
{
    if (parent_->IsTopPositionFilled()) {
        AddDoor();
    }
}

} // namespace objects

} // namespace nextfloor
