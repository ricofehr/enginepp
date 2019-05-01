/**
 *  @file run_command.cc
 *  @brief Run Command class file, TODO: Implement Run Execute action and Undo Part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/run_command.h"

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
 *  Execute Run action on actor Game Object
 *  @param actor Game Object targetted
 */
void RunCommand::execute(nextfloor::universe::Model3D* actor)
{
    ;
}

} // namespace commands

} // namespace universe

} // namespace nextfloor