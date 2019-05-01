/**
 *  @file moveleft_command.cc
 *  @brief MoveLeft Command class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/moveleft_command.h"

#include <glm/glm.hpp>

#include "nextfloor/core/global_timer.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::universe
 *  @brief World elements
 */
namespace universe {

/**
 *  @namespace nextfloor::universe::commands
 *  @brief commands event
 */
namespace commands {

/**
 *  Execute Move Left action on actor Game Object
 *  @param actor Game Object targetted
 */
void MoveLeftCommand::execute(nextfloor::universe::Model3D* actor)
{
    using nextfloor::core::GlobalTimer;

    /* Left vector */
    glm::vec3 left = -glm::cross(actor->direction(), actor->head());
    actor->set_move(left * (float)GlobalTimer::sDeltaTime * actor->get_speed());
}

} // namespace commands

} // namespace universe

} // namespace nextfloor