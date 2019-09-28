/**
 *  @file moveleft_action.cc
 *  @brief MoveLeft Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/moveleft_action.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace actions {

void MoveLeftAction::execute(nextfloor::objects::Mesh* actor, double elapsed_time)
{
    glm::vec3 movement = actor->movement();
    glm::vec3 head = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        if (camera != nullptr) {
            movement = actor->camera()->direction();
            if (actor->IsPlayer()) {
                movement *= elapsed_time;
            }
            head = camera->head();
        }
    }

    /* Left vector */
    glm::vec3 left_movement = -glm::cross(movement, head);
    actor->set_movement(left_movement);
}

}  // namespace actions

}  // namespace nextfloor
