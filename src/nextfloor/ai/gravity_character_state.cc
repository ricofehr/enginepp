/**
 *  @file gravity_character_state.cc
 *  @brief GravityCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/gravity_character_state.h"

#include <glm/glm.hpp>
#include <iostream>

namespace nextfloor {

namespace ai {

void GravityCharacterState::Enter(nextfloor::character::Character* actor)
{
    assert(actor != nullptr);

    float location_y = actor->location().y;
    current_y_stage_ = location_y;

    is_interruptible_ = false;
}

void GravityCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    assert(actor != nullptr);

    float location_y = actor->location().y;
    glm::vec3 movement = actor->movement();
    movement.y = -(kDistance / kDuration) * static_cast<float>(elapsed_time);
    if (!first_execution_) {
        if (location_y >= current_y_stage_) {
            is_finished_ = true;
            movement.y = 0.0f;
        } else {
            movement.y *= 1.5;
            movement.x /= 2;
            movement.z /= 2;
        }
    }
    actor->set_movement(movement);

    current_y_stage_ = location_y;
    first_execution_ = false;
}

void GravityCharacterState::Exit(nextfloor::character::Character* actor)
{
    is_interruptible_ = true;
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor