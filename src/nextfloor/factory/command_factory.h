/**
 *  @file command_factory.h
 *  @brief Abstract Factory Class Implementation for action part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_COMMANDFACTORY_H_
#define NEXTFLOOR_FACTORY_COMMANDFACTORY_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace factory {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for player and NPJ actions
 */
class CommandFactory {

public:
    virtual ~CommandFactory() = default;

    virtual std::unique_ptr<nextfloor::actions::Action> MakeFireAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeJumpAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveDownAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveLeftAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveRightAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveUpAction() = 0;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeRunAction() = 0;

protected:
    CommandFactory() = default;

    CommandFactory(CommandFactory&&) = default;
    CommandFactory& operator=(CommandFactory&&) = default;
    CommandFactory(const CommandFactory&) = default;
    CommandFactory& operator=(const CommandFactory&) = default;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_COMMANDFACTORY_H_
