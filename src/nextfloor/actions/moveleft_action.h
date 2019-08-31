/**
 *  @file fire_action.h
 *  @brief Fire action for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_
#define NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveLeftAction
 *  @brief Implements Action (Command Pattern), generate Move Left action for any object
 */
class MoveLeftAction : public Action {

public:
    MoveLeftAction() = default;

    MoveLeftAction(MoveLeftAction&&) = default;
    MoveLeftAction& operator=(MoveLeftAction&&) = default;
    MoveLeftAction(const MoveLeftAction&) = default;
    MoveLeftAction& operator=(const MoveLeftAction&) = default;

    virtual ~MoveLeftAction() override = default;

    void execute(nextfloor::objects::Mesh* actor) override final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_
