/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/character/player.h"

#include <glm/glm.hpp>
#include "nextfloor/playground/ground.h"

namespace nextfloor {

namespace character {

Player::Player(std::unique_ptr<nextfloor::mesh::Border> border, std::unique_ptr<Camera> camera)
{
    border_ = std::move(border);
    camera->set_owner(this);
    camera_ = std::move(camera);
}

void Player::MoveLocation()
{
    assert(border_ != nullptr);

    border_->ComputeNewLocation();
    parent_ = parent_->UpdateChildPlacement(this);

    /* After each step, get back control to HID Device */
    reset_movement();
}

Camera* Player::camera() const
{
    assert(camera_ != nullptr);
    return camera_.get();
}

}  // namespace character

}  // namespace nextfloor
