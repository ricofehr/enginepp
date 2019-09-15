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
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(0.2f));
    camera_ = CommonServices::getFactory()->MakeCamera(this);
}

void Player::Move() noexcept
{
    if (IsMoved()) {
        border_->ComputeNewLocation();

        if (parent_->IsInside(this)) {
            parent_->UpdateChildPlacement(this);
        }
        else {
            parent_ = parent_->TransfertChildToNeighbor(this);
        }
    }

    set_movement(glm::vec3(0.0f));
}

}  // namespace objects

}  // namespace nextfloor
