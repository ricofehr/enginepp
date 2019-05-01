/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_FIRECOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_FIRECOMMAND_H_

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
 *  @class FireCommand
 *  @brief Implements the Command, generate Fire action for an object
 */
class FireCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    FireCommand() = default;

    /**
     *  Default Move constructor
     */
    FireCommand(FireCommand&&) = default;

    /**
     *  Default Move assignment
     */
    FireCommand& operator=(FireCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    FireCommand(const FireCommand&) = default;

    /**
     *  Default Copy assignment
     */
    FireCommand& operator=(const FireCommand&) = default;

    /**
     *  Default destructor
     */
    ~FireCommand() override = default;

    /**
     *  Execute Fire action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_FIRECOMMAND_H_
