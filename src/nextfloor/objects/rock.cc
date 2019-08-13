/**
 *  @file brick.cc
 *  @brief Brick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/rock.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Rock::Rock(glm::vec3 location, float scale)
{
    using nextfloor::core::CommonServices;

    polygons_.push_back(CommonServices::getFactory()->MakeCube(location, glm::vec3(scale)));
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(scale));
    renderer_ = CommonServices::getFactory()->MakeCubeRenderer(kTEXTURE);

    set_movement(glm::vec3(0.3,0.01,0.05));
}

} // namespace objects

} // namespace nextfloor
