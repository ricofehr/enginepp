/**
 *  @file moveleft_action.cc
 *  @brief MoveLeft Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/moveleft_action.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace actions {

void MoveLeftAction::execute(nextfloor::gameplay::Character* actor, double elapsed_time)
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

    /* Left vector */
    glm::vec3 left_movement = -glm::cross(movement, head);
    actor->set_movement(left_movement);
}

}  // namespace actions

}  // namespace nextfloor
