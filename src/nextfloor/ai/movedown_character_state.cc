/**
 *  @file movedown_character_state.cc
 *  @brief MoveDownCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/movedown_character_state.h"


namespace nextfloor {

namespace ai {

void MoveDownCharacterState::Enter(nextfloor::character::Character* actor)
{
    is_finished_ = false;
}

void MoveDownCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction() * kMoveFactor;
        if (!actor->is_flying()) {
            movement.y = 0.0f;
        }
        movement *= elapsed_time;
    }

    actor->set_movement(-movement);
    is_finished_ = true;
}

void MoveDownCharacterState::Exit(nextfloor::character::Character* actor)
{
    if (actor->is_flying()) {
        owner_->Idle();
    }
    else {
        owner_->ApplyGravity();
    }
}

}  // namespace ai

}  // namespace nextfloor