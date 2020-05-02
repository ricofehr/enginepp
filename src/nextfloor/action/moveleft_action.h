/**
 *  @file moveleft_action.h
 *  @brief Move Left action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_MOVELEFTACTION_H_
#define NEXTFLOOR_ACTION_MOVELEFTACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace action {

/**
 *  @class MoveLeftAction
 *  @brief Implements Action (Command Pattern), generate Move Left action for any character
 */
class MoveLeftAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor) final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_MOVELEFTACTION_H_
