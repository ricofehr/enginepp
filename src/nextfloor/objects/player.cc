/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/player.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Player::Player(glm::vec3 location)
{
    using nextfloor::core::CommonServices;
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(1.0f));
    camera_ = CommonServices::getFactory()->MakeCamera(this);
}

} // namespace objects

} // namespace nextfloor
