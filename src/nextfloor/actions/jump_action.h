/**
 *  @file jump_action.h
 *  @brief Jump Action Heder file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_JUMPACTION_H_
#define NEXTFLOOR_ACTIONS_JUMPACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class JumpAction
 *  @brief Implements Action (Command Pattern), generate Jump action for any character
 */
class JumpAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor, double elapsed_time) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_JUMPACTION_H_
