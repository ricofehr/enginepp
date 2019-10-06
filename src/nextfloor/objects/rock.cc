/**
 *  @file brick.cc
 *  @brief Brick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/rock.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Rock::Rock(const glm::vec3& location, float scale, const MeshFactory& factory)
{
    polygons_.push_back(factory.MakeCube(location, glm::vec3(scale), kTEXTURE));
    border_ = factory.MakeBorder(location, glm::vec3(scale));

    set_movement(glm::vec3(0.3, 0.01, 0.05));
}

}  // namespace objects

}  // namespace nextfloor
