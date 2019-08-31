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
    ActionFactory() = default;

    ActionFactory(ActionFactory&&) = default;
    ActionFactory& operator=(ActionFactory&&) = default;
    ActionFactory(const ActionFactory&) = default;
    ActionFactory& operator=(const ActionFactory&) = default;

    virtual ~ActionFactory() override = default;

    virtual std::unique_ptr<nextfloor::actions::Action> MakeFireAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeJumpAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveDownAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveLeftAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveRightAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeMoveUpAction() override;
    virtual std::unique_ptr<nextfloor::actions::Action> MakeRunAction() override;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_ACTIONFACTORY_H_
