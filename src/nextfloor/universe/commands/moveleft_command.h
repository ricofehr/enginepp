/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_MOVELEFTCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_MOVELEFTCOMMAND_H_

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
 *  @class MoveLeftCommand
 *  @brief Implements the Command, generate Move action for an object
 */
class MoveLeftCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    MoveLeftCommand() = default;

    /**
     *  Default Move constructor
     */
    MoveLeftCommand(MoveLeftCommand&&) = default;

    /**
     *  Default Move assignment
     */
    MoveLeftCommand& operator=(MoveLeftCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    MoveLeftCommand(const MoveLeftCommand&) = default;

    /**
     *  Default Copy assignment
     */
    MoveLeftCommand& operator=(const MoveLeftCommand&) = default;

    /**
     *  Default destructor
     */
    ~MoveLeftCommand() override = default;

    /**
     *  Execute Move Left action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_MOVELEFTCOMMAND_H_
