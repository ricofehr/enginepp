/**
 *  @file movedown_command.cc
 *  @brief MoveDown Command class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/movedown_command.h"

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
 *  Execute Move Bottom action on actor Game Object
 *  @param actor Game Object targetted
 */
void MoveDownCommand::execute(nextfloor::universe::Model3D* actor)
{
    using nextfloor::core::GlobalTimer;

    actor->set_move(-actor->direction() * (float)GlobalTimer::sDeltaTime * actor->get_speed());
}

} // namespace commands

} // namespace universe

} // namespace nextfloor