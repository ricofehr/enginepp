/**
 *  @file jump_action.cc
 *  @brief Jump Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/jump_action.h"

#include <cassert>
#include <glm/glm.hpp>

namespace nextfloor {

namespace action {

void JumpAction::execute(nextfloor::character::Character* actor)
{
    assert(actor != nullptr);
    actor->Jump();
}

}  // namespace action

}  // namespace nextfloor
