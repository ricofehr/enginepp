/**
 *  @file run_action.h
 *  @brief Run action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_RUNACTION_H_
#define NEXTFLOOR_ACTION_RUNACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace action {

/**
 *  @class MoveRunAction
 *  @brief Implements Action (Command Pattern), generate Run action for any character
 */
class RunAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor, double elapsed_time) final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_RUNACTION_H_
