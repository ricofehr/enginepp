/**
 *  @file run_command.h
 *  @brief Run command for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_COMMANDS_RUNCOMMAND_H_
#define NEXTFLOOR_UNIVERSE_COMMANDS_RUNCOMMAND_H_

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
 *  @class RunCommand
 *  @brief Implements the Command, generate Run action for an object
 */
class RunCommand : public Command {

public:

    /**
     *  Default Constructor
     */
    RunCommand() = default;

    /**
     *  Default Move constructor
     */
    RunCommand(RunCommand&&) = default;

    /**
     *  Default Move assignment
     */
    RunCommand& operator=(RunCommand&&) = default;

    /**
     *  Default Copy constructor
     */
    RunCommand(const RunCommand&) = default;

    /**
     *  Default Copy assignment
     */
    RunCommand& operator=(const RunCommand&) = default;

    /**
     *  Default destructor
     */
    ~RunCommand() override = default;

    /**
     *  Execute Run action on actor Game Object
     *  @param actor Game Object targetted
     */
    void execute(nextfloor::universe::Model3D* actor) override final;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_COMMANDS_RUNCOMMAND_H_
