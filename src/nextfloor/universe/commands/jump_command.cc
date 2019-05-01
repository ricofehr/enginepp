/**
 *  @file moveup_command.cc
 *  @brief MOveUp Command class file, TODO: Implement Jump Execute action and Undo Part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/jump_command.h"

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
 *  Execute Jump action on actor Game Object
 *  @param actor Game Object targetted
 */
void JumpCommand::execute(nextfloor::universe::Model3D* actor)
{
    ;
}

} // namespace commands

} // namespace universe

} // namespace nextfloor