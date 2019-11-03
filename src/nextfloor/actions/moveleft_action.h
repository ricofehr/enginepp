/**
 *  @file moveleft_action.h
 *  @brief Move Left action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_
#define NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveLeftAction
 *  @brief Implements Action (Command Pattern), generate Move Left action for any character
 */
class MoveLeftAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor, double elapsed_time) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVELEFTACTION_H_
