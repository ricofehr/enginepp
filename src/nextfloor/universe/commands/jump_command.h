/**
 *  @file fire_command.h
 *  @brief Fire command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_JUMPCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_JUMPCOMMAND_H_

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
 *  @class JumpCommand
 *  @brief Implements the Command, generate Jump action for an object
 */
class JumpCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    JumpCommand() = default;

    /**
     *  Default Move constructor
     */
    JumpCommand(JumpCommand&&) = default;

    /**
     *  Default Move assignment
     */
    JumpCommand& operator=(JumpCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    JumpCommand(const JumpCommand&) = default;

    /**
     *  Default Copy assignment
     */
    JumpCommand& operator=(const JumpCommand&) = default;

    /**
     *  Default destructor
     */
    ~JumpCommand() override = default;

    /**
     *  Execute Jump action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_JUMPCOMMAND_H_
