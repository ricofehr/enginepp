/**
 *  @file moveleft_action.cc
 *  @brief MoveLeft Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/moveleft_action.h"

#include <cassert>
#include <glm/glm.hpp>

namespace nextfloor {

namespace action {

void MoveLeftAction::execute(nextfloor::character::Character* actor)
{
    assert(actor != nullptr);
    actor->MoveLeft();
}

}  // namespace action

}  // namespace nextfloor
