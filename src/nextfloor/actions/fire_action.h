/**
 *  @file fire_action.h
 *  @brief Fire action Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_FIREACTION_H_
#define NEXTFLOOR_ACTIONS_FIREACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class FireAction
 *  @brief Implements Action (Command Pattern), generate Fire action for any character
 */
class FireAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::gameplay::Character* actor, double elapsed_time) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_FIREACTION_H_
