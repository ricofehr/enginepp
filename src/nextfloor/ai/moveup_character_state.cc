/**
 *  @file moveup_character_state.cc
 *  @brief MoveUpCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/moveup_character_state.h"

#include <iostream>

namespace nextfloor {

namespace ai {

void MoveUpCharacterState::Enter(nextfloor::character::Character* actor)
{
    actor->reset_movement();
}

void MoveUpCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    assert(actor != nullptr);

    glm::vec3 movement = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->owner_movement();
        movement *= elapsed_time;
    }

    actor->set_movement(movement);
    is_finished_ = true;
}

void MoveUpCharacterState::Exit(nextfloor::character::Character* actor)
{
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor
