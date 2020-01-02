/**
 *  @file action_factory.h
 *  @brief ActionFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_FACTORY_H_
#define NEXTFLOOR_GAMEPLAY_FACTORY_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class ActionFactory
 *  @brief Abstract Factory for character actions
 */
class ActionFactory {

public:
    virtual ~ActionFactory() = default;

    virtual std::unique_ptr<Action> MakeMoveDownAction() const = 0;
    virtual std::unique_ptr<Action> MakeMoveLeftAction() const = 0;
    virtual std::unique_ptr<Action> MakeMoveRightAction() const = 0;
    virtual std::unique_ptr<Action> MakeMoveUpAction() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_ACTIONFACTORY_H_
