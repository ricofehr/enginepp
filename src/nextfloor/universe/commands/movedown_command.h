/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_MOVEDOWNCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_MOVEDOWNCOMMAND_H_

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
 *  @class MoveDownCommand
 *  @brief Implements the Command, generate Move action for an object
 */
class MoveDownCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    MoveDownCommand() = default;

    /**
     *  Default Move constructor
     */
    MoveDownCommand(MoveDownCommand&&) = default;

    /**
     *  Default Move assignment
     */
    MoveDownCommand& operator=(MoveDownCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    MoveDownCommand(const MoveDownCommand&) = default;

    /**
     *  Default Copy assignment
     */
    MoveDownCommand& operator=(const MoveDownCommand&) = default;

    /**
     *  Default destructor
     */
    ~MoveDownCommand() override = default;

    /**
     *  Execute Move Bottom action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_MOVEDOWNCOMMAND_H_
