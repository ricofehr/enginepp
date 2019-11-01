/**
 *  @file moveup_action.cc
 *  @brief MOveUp Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/moveup_action.h"

namespace nextfloor {

namespace actions {

void MoveUpAction::execute(nextfloor::gameplay::Character* actor, double elapsed_time)
{
    assert(actor != nullptr);

    glm::vec3 movement = actor->movement();
    if (actor->IsCamera()) {
        auto camera = actor->camera();
        movement = camera->direction();
        if (elapsed_time != kFrameElapsedTime) {
            movement *= elapsed_time;
        }
    }

    actor->set_movement(movement);
}

}  // namespace actions

}  // namespace nextfloor
