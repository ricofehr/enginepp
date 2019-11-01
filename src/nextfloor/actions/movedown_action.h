/**
 *  @file movedown_action.h
 *  @brief MoveDown Action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_
#define NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveDownAction
 *  @brief Implements Action (Command Pattern), generate Move Down action for any character
 */
class MoveDownAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::gameplay::Character* actor, double elapsed_time) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVEDOWNACTION_H_
