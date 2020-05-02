/**
 *  @file moveright_action.cc
 *  @brief MoveRight Action class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/moveright_action.h"

#include <cassert>
#include <glm/glm.hpp>

namespace nextfloor {

namespace action {

void MoveRightAction::execute(nextfloor::character::Character* actor)
{
    assert(actor != nullptr);
    actor->MoveRight();
}

}  // namespace action

}  // namespace nextfloor
