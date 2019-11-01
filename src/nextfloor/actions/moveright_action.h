/**
 *  @file moveright_action.h
 *  @brief MoveRight action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_
#define NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class MoveRightAction
 *  @brief Implements Action (Command Pattern), generate Move Right action for any character
 */
class MoveRightAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::gameplay::Character* actor, double elapsed_time) final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_MOVERIGHTACTION_H_
