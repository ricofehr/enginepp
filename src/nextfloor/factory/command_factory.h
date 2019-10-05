/**
 *  @file command_factory.h
 *  @brief Abstract Factory Class Implementation for action part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_COMMANDFACTORY_H_
#define NEXTFLOOR_FACTORY_COMMANDFACTORY_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace factory {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for player and NPJ actions
 */
class CommandFactory {

public:
    virtual ~CommandFactory() = default;

    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeFireAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeJumpAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveDownAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveLeftAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveRightAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveUpAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeRunAction() const = 0;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_COMMANDFACTORY_H_
