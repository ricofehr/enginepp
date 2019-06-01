/**
 *  @file moveright_command.cc
 *  @brief MoveRight Command class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/universe/commands/moveright_command.h"

#include <glm/glm.hpp>

#include "nextfloor/core/common_services.h"

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
 *  Execute Move Right action on actor Game Object
 *  @param actor Game Object targetted
 */
void MoveRightCommand::execute(nextfloor::universe::Model3D* actor)
{
    using nextfloor::core::CommonServices;

    /* Right vector */
    glm::vec3 right = glm::cross(actor->direction(), actor->head());
    actor->set_move(right * CommonServices::getTimer()->getDeltaTimeSinceLastLoop() * actor->get_speed());
}

} // namespace commands

} // namespace universe

} // namespace nextfloor
