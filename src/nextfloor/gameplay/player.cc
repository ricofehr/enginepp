/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/player.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

Player::Player(const glm::vec3& location,
               std::unique_ptr<nextfloor::objects::Border> border,
               std::unique_ptr<nextfloor::objects::Camera> camera)
{
    border_ = std::move(border);
    camera->set_owner(this);
    camera_ = std::move(camera);
}

void Player::MoveLocation()
{
    border_->ComputeNewLocation();
    set_movement(glm::vec3(0.0f));
}

}  // namespace gameplay

}  // namespace nextfloor
