/**
 *  @file fire_action.h
 *  @brief Fire action for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVEUPACTION_H_
#define NEXTFLOOR_ACTIONS_MOVEUPACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveUpAction
 *  @brief Implements Action (Command Pattern), generate Move Up action for any object
 */
class MoveUpAction : public Action {

public:
    MoveUpAction() = default;

    MoveUpAction(MoveUpAction&&) = default;
    MoveUpAction& operator=(MoveUpAction&&) = default;
    MoveUpAction(const MoveUpAction&) = default;
    MoveUpAction& operator=(const MoveUpAction&) = default;

    virtual ~MoveUpAction() override = default;

    void execute(nextfloor::objects::Mesh* actor) override final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVEUPACTION_H_
