/**
 *  @file roof.cc
 *  @brief Roof class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/roof.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Roof::Roof(const glm::vec3& location, const glm::vec3& scale)
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    border_ = factory->MakeBorder(location, glm::vec3(scale));
    brick_dimension_ = glm::vec3(kBRICK_WIDTH, kBRICK_HEIGHT, kBRICK_DEPTH);
    bricks_count_ = border_->dimension() / brick_dimension_;
    AddBricks(location - scale, location + scale);
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

void Roof::PrepareDraw(const Camera& active_camera)
{
    if (parent_->IsTopPositionFilled()) {
        AddDoor();
    }

    Wall::PrepareDraw(active_camera);
}

}  // namespace objects

}  // namespace nextfloor
