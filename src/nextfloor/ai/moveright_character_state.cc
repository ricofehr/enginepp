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
    is_finished_ = false;
}

void MoveRightCharacterState::Execute(nextfloor::character::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    auto head = movement;
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction() * kMoveFactor;
        if (!actor->is_flying()) {
            movement.y = 0.0f;
        }
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
    if (actor->is_flying()) {
        owner_->Idle();
    }
    else {
        owner_->ApplyGravity();
    }
}

}  // namespace ai

}  // namespace nextfloor
