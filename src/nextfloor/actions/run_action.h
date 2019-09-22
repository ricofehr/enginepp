/**
 *  @file run_action.h
 *  @brief Run action for any object
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_RUNACTION_H_
#define NEXTFLOOR_ACTIONS_RUNACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveRunAction
 *  @brief Implements Action (Command Pattern), generate Run action for any object
 */
class RunAction : public Action {

public:
    void execute(nextfloor::objects::Mesh* actor) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_RUNACTION_H_
