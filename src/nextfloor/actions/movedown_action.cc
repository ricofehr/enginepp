/**
 *  @file movedown_action.cc
 *  @brief MoveDown Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/movedown_action.h"

#include <cassert>

namespace nextfloor {

namespace actions {

void MoveDownAction::execute(nextfloor::gameplay::Character* actor, double elapsed_time)
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

    actor->set_movement(-movement);
}

}  // namespace actions

}  // namespace nextfloor
