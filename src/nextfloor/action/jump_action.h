/**
 *  @file jump_action.h
 *  @brief Jump Action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_JUMPACTION_H_
#define NEXTFLOOR_ACTION_JUMPACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace action {

/**
 *  @class JumpAction
 *  @brief Implements Action (Command Pattern), generate Jump action for any character
 */
class JumpAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor) final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_JUMPACTION_H_
