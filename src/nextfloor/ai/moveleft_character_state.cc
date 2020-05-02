/**
 *  @file moveleft_character_state.cc
 *  @brief MoveLeftCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/moveleft_character_state.h"


namespace nextfloor {

namespace ai {

void MoveLeftCharacterState::Enter(nextfloor::character::Character* actor)
{
    actor->reset_movement();
}

void MoveLeftCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    auto head = movement;
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        movement *= elapsed_time;
        head = camera->head();
    }

    /* Left vector: perpendicular (opposite) to both direction and head vector */
    glm::vec3 left_movement = -glm::cross(movement, head);
    actor->set_movement(left_movement);
    is_finished_ = true;
}

void MoveLeftCharacterState::Exit(nextfloor::character::Character* actor)
{
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor
