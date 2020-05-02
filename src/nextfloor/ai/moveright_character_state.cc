/**
 *  @file moveright_character_state.cc
 *  @brief MoveRightCharacterState class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/moveright_character_state.h"


namespace nextfloor {

namespace ai {

void MoveRightCharacterState::Enter(nextfloor::character::Character* actor)
{
    actor->reset_movement();
}

void MoveRightCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    auto head = movement;
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        movement *= elapsed_time;
        head = camera->head();
    }

    /* Right vector: perpendicular to both direction and head vector */
    glm::vec3 right_movement = glm::cross(movement, head);
    actor->set_movement(right_movement);
    is_finished_ = true;
}

void MoveRightCharacterState::Exit(nextfloor::character::Character* actor)
{
    owner_->Idle();
}

}  // namespace ai

}  // namespace nextfloor
