/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/player.h"

namespace nextfloor {

namespace gameplay {

Player::Player(const glm::vec3& location, std::unique_ptr<nextfloor::objects::Border> border, std::unique_ptr<Camera> camera)
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

Camera* Player::camera() const
{
    assert(camera_ != nullptr);
    return camera_.get();
}

}  // namespace gameplay

}  // namespace nextfloor
