/**
 *  @file command.h
 *  @brief Command class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_COMMANDS_COMMAND_H_
#define NEXTFLOOR_COMMANDS_COMMAND_H_

#include "nextfloor/universe/model3d.h"

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
 *  @class Command
 *  @brief Command interface
 */
class Command {

public:
    /**
     *  Default Move constructor
     */
    Command(Command&&) = default;

    /**
     *  Default Move assignment
     */
    Command& operator=(Command&&) = default;

    /**
     *  Default Copy constructor
     */
    Command(const Command&) = default;

    /**
     *  Default Copy assignment
     */
    Command& operator=(const Command&) = default;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~Command() = default;

    /**
     * [execute description]
     */
    virtual void execute(nextfloor::universe::Model3D* actor) = 0;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    Command() = default;

};

} // namespace commands

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_COMMANDS_COMMAND_H_
