/**
 *  @file moveright_action.cc
 *  @brief MoveRight Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/moveright_action.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace actions {

void MoveRightAction::execute(nextfloor::gameplay::Character* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    glm::vec3 head = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        if (actor->IsPlayer()) {
            movement *= elapsed_time;
        }
        head = camera->head();
    }

    /* Right vector */
    glm::vec3 right_movement = glm::cross(movement, head);
    actor->set_movement(right_movement);
}

}  // namespace actions

}  // namespace nextfloor
