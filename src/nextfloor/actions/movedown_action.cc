/**
 *  @file movedown_action.cc
 *  @brief MoveDown Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/movedown_action.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace actions {

void MoveDownAction::execute(nextfloor::objects::Mesh* actor)
{
    glm::vec3 movement = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        if (camera != nullptr) {
            movement = actor->camera()->direction();
            if (actor->IsPlayer()) {
                using nextfloor::core::CommonServices;
                movement *= CommonServices::getTimer().getDeltaTimeSinceLastLoop();
            }
        }
    }

    actor->set_movement(-movement);
}

}  // namespace actions

}  // namespace nextfloor
