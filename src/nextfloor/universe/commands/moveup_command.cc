/**
 *  @file moveup_command.cc
 *  @brief MOveUp Command class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/moveup_command.h"

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
 *  Execute Move Up action on actor Game Object
 *  @param actor Game Object targetted
 */
void MoveUpCommand::execute(nextfloor::universe::Model3D* actor)
{
    using nextfloor::core::GlobalTimer;

    actor->set_move(actor->direction() * (float)GlobalTimer::sDeltaTime * actor->get_speed());
}

} // namespace commands

} // namespace universe

} // namespace nextfloor