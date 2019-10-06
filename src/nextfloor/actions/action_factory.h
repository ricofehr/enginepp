/**
 *  @file action_factory.h
 *  @brief Abstract Factory Class Implementation for action part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_FACTORY_H_
#define NEXTFLOOR_ACTIONS_FACTORY_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for player and NPJ actions
 */
class ActionFactory {

public:
    virtual ~ActionFactory() = default;

    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeFireAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeJumpAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveDownAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveLeftAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveRightAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeMoveUpAction() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::Action> MakeRunAction() const = 0;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_ACTIONFACTORY_H_
