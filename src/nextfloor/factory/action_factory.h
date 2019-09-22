/**
 *  @file action_factory.h
 *  @brief Abstract Factory Class Implementation for action part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_ACTIONFACTORY_H_
#define NEXTFLOOR_FACTORY_ACTIONFACTORY_H_

#include "nextfloor/factory/command_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for player and NPJ actions
 */
class ActionFactory : public CommandFactory {

public:
    std::unique_ptr<nextfloor::actions::Action> MakeFireAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeJumpAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeMoveDownAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeMoveLeftAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeMoveRightAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeMoveUpAction() const final;
    std::unique_ptr<nextfloor::actions::Action> MakeRunAction() const final;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_ACTIONFACTORY_H_
