/**
 *  @file fire_command.cc
 *  @brief Fire Command class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/fire_command.h"

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
 *  Execute Fire action on actor Game Object
 *  @param actor Game Object targetted
 */
void FireCommand::execute(nextfloor::universe::Model3D* actor)
{
    ;
}

} // namespace commands

} // namespace universe

} // namespace nextfloor