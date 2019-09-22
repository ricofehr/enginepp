/**
 *  @file moveright_action.cc
 *  @brief MoveRight Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/moveright_action.h"

#include <glm/glm.hpp>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace actions {

void MoveRightAction::execute(nextfloor::objects::Mesh* actor)
{
    glm::vec3 movement = actor->movement();
    glm::vec3 head = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        if (camera != nullptr) {
            movement = actor->camera()->direction();
            if (actor->IsPlayer()) {
                using nextfloor::core::CommonServices;
                movement *= CommonServices::getTimer().getDeltaTimeSinceLastLoop();
            }
            head = camera->head();
        }
    }

    /* Right vector */
    glm::vec3 right_movement = glm::cross(movement, head);
    actor->set_movement(right_movement);
}

}  // namespace actions

}  // namespace nextfloor
