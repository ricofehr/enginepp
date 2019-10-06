/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/player.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

Player::Player(const glm::vec3& location, const nextfloor::objects::MeshFactory& mesh_factory)
{
    border_ = mesh_factory.MakeBorder(location, glm::vec3(0.4f));
    camera_ = mesh_factory.MakeCamera(this);
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

}  // namespace gameplay

}  // namespace nextfloor
