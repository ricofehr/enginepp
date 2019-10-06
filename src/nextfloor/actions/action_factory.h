/**
 *  @file action_factory.h
 *  @brief Abstract Factory Class Implementation for action part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_ACTIONFACTORY_H_
#define NEXTFLOOR_ACTIONS_ACTIONFACTORY_H_

#include "nextfloor/actions/factory.h"

namespace nextfloor {

namespace actions {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for player and NPJ actions
 */
class ActionFactory : public Factory {

public:
    std::unique_ptr<nextfloor::gameplay::Action> MakeFireAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeJumpAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveDownAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveLeftAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveRightAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveUpAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeRunAction() const final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_ACTIONFACTORY_H_
