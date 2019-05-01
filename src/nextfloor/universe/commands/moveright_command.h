/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_MOVERIGHTCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_MOVERIGHTCOMMAND_H_

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
 *  @class MoveRightCommand
 *  @brief Implements the Command, generate Move action for an object
 */
class MoveRightCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    MoveRightCommand() = default;

    /**
     *  Default Move constructor
     */
    MoveRightCommand(MoveRightCommand&&) = default;

    /**
     *  Default Move assignment
     */
    MoveRightCommand& operator=(MoveRightCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    MoveRightCommand(const MoveRightCommand&) = default;

    /**
     *  Default Copy assignment
     */
    MoveRightCommand& operator=(const MoveRightCommand&) = default;

    /**
     *  Default destructor
     */
    ~MoveRightCommand() override = default;

    /**
     *  Execute Move Right action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_MOVERIGHTCOMMAND_H_
