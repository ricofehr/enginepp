/**
 *  @file fire_action.h
 *  @brief Fire action for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_
#define NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveDownAction
 *  @brief Implements Action (Command Pattern), generate Move Down action for any object
 */
class MoveDownAction : public Action {

public:
    MoveDownAction() = default;

    MoveDownAction(MoveDownAction&&) = default;
    MoveDownAction& operator=(MoveDownAction&&) = default;
    MoveDownAction(const MoveDownAction&) = default;
    MoveDownAction& operator=(const MoveDownAction&) = default;

    virtual ~MoveDownAction() override = default;

    void execute(nextfloor::objects::Mesh* actor) override final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_
