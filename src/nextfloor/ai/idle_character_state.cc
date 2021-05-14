/**
 *  @file idle_character_state.cc
 *  @brief IdleCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/idle_character_state.h"


namespace nextfloor {

namespace ai {

void IdleCharacterState::Enter(nextfloor::character::Character* actor)
{
    //actor->reset_movement();
    is_finished_ = false;
}

void IdleCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    actor->reset_movement();
    is_finished_ = true;
}

void IdleCharacterState::Exit(nextfloor::character::Character* actor)
{
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor