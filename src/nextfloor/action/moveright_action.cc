/**
 *  @file moveright_action.cc
 *  @brief MoveRight Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/moveright_action.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace action {

void MoveRightAction::execute(nextfloor::character::Character* actor, double elapsed_time)
{
    assert(actor != nullptr);

    glm::vec3 movement = actor->movement();
    auto head = movement;
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        if (elapsed_time != kFrameElapsedTime) {
            movement *= elapsed_time;
        }
        head = camera->head();
    }

    /* Right vector */
    glm::vec3 right_movement = glm::cross(movement, head);
    actor->set_movement(right_movement);
}

}  // namespace action

}  // namespace nextfloor
