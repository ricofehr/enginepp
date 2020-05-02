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
    actor->reset_movement();
}

void MoveDownCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        movement *= elapsed_time;
    }

    actor->set_movement(-movement);
    is_finished_ = true;
}

void MoveDownCharacterState::Exit(nextfloor::character::Character* actor)
{
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor