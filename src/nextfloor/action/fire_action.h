/**
 *  @file fire_action.h
 *  @brief Fire action Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_FIREACTION_H_
#define NEXTFLOOR_ACTION_FIREACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace action {

/**
 *  @class FireAction
 *  @brief Implements Action (Command Pattern), generate Fire action for any character
 */
class FireAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor, double elapsed_time) final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_FIREACTION_H_
