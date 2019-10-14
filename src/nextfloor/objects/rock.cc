/**
 *  @file brick.cc
 *  @brief Brick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/rock.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Rock::Rock(std::unique_ptr<Border> border,
           std::vector<std::unique_ptr<nextfloor::polygons::Polygon>> rock,
           const glm::vec3& movement)
{
    polygons_ = std::move(rock);
    border_ = std::move(border);

    set_movement(movement);
}

}  // namespace objects

}  // namespace nextfloor
