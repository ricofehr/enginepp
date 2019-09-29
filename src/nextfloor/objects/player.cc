/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/player.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Player::Player(const glm::vec3& location)
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    border_ = factory->MakeBorder(location, glm::vec3(0.4f));
    camera_ = factory->MakeCamera(this);
}

void Player::Move()
{
    if (IsMoved()) {
        border_->ComputeNewLocation();

        if (parent_->IsInside(*this)) {
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
