/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_MOVEUPCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_MOVEUPCOMMAND_H_

#include "nextfloor/universe/commands/command.h"

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
 *  @class MoveUpCommand
 *  @brief Implements the Command, generate Move action for an object
 */
class MoveUpCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    MoveUpCommand() = default;

    /**
     *  Default Move constructor
     */
    MoveUpCommand(MoveUpCommand&&) = default;

    /**
     *  Default Move assignment
     */
    MoveUpCommand& operator=(MoveUpCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    MoveUpCommand(const MoveUpCommand&) = default;

    /**
     *  Default Copy assignment
     */
    MoveUpCommand& operator=(const MoveUpCommand&) = default;

    /**
     *  Default destructor
     */
    ~MoveUpCommand() override = default;

    /**
     *  Execute Move Up action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_MOVEUPCOMMAND_H_
