/**
 *  @file player.cc
 *  @brief Player class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/character/player.h"

#include <cassert>
#include <utility>

namespace nextfloor {

namespace character {

Player::Player(std::unique_ptr<nextfloor::mesh::Border> border, std::unique_ptr<Camera> camera, std::unique_ptr<FSM> fsm)
{
    border_ = std::move(border);
    camera->set_owner(this);
    camera_ = std::move(camera);
    fsm_ = std::move(fsm);
    fsm_->set_owner(this);
}

void Player::MoveLocation()
{
    assert(border_ != nullptr);

    border_->ComputeNewLocation();
    parent_ = parent_->UpdateChildPlacement(this);
}

Camera* Player::camera() const
{
    assert(camera_ != nullptr);
    return camera_.get();
}

}  // namespace character

}  // namespace nextfloor
