/**
 *  @file fire_action.h
 *  @brief Fire action for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_
#define NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveRightAction
 *  @brief Implements Action (Command Pattern), generate Move Right action for any object
 */
class MoveRightAction : public Action {

public:
    void execute(nextfloor::objects::Mesh* actor) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_
