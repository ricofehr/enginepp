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
    RunAction() = default;

    RunAction(RunAction&&) = default;
    RunAction& operator=(RunAction&&) = default;
    RunAction(const RunAction&) = default;
    RunAction& operator=(const RunAction&) = default;

    virtual ~RunAction() override = default;

    void execute(nextfloor::objects::Mesh* actor) override final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_RUNACTION_H_
