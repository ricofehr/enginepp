/**
 *  @file movedown_action.cc
 *  @brief MoveDown Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/movedown_action.h"

#include <cassert>
#include <glm/glm.hpp>

namespace nextfloor {

namespace action {

void MoveDownAction::execute(nextfloor::character::Character* actor)
{
    assert(actor != nullptr);
    actor->MoveDown();
}

}  // namespace action

}  // namespace nextfloor
