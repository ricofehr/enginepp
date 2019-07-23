#include "nextfloor/objects/wall.h"

#include <memory>
#include <iostream>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

void Wall::AddBricks(glm::vec3 firstpoint, glm::vec3 lastpoint) noexcept
{
    using nextfloor::core::CommonServices;

    auto padding = brick_dimension_/2.0f;
    firstpoint += padding;
    lastpoint -= padding;

    //bricks_count_ = glm::ivec3(0,0,0);
    for (float x = firstpoint.x; x <= lastpoint.x; x += brick_dimension_.x) {
        ++bricks_count_.x;
        for (float y = firstpoint.y; y <= lastpoint.y; y += brick_dimension_.y) {
            for (float z = firstpoint.z; z <= lastpoint.z; z += brick_dimension_.z) {
                auto brick_location = glm::vec3(x,y,z);
                add_child(CommonServices::getFactory()->MakeWallBrick(brick_location, brick_dimension_/2.0f, texture_file()));
                //auto brick = CommonServices::getFactory()->MakeCube(brick_location, brick_dimension_/2.0f);
                //meshes_.push_back(std::move(brick));
            }
        }
    }
}

} // namespace objects

} // namespace nextfloor
