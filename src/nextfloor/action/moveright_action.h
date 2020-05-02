/**
 *  @file moveright_action.h
 *  @brief MoveRight action Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_MOVERIGHTACTION_H_
#define NEXTFLOOR_ACTION_MOVERIGHTACTION_H_

#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace action {

/**
 *  @class MoveRightAction
 *  @brief Implements Action (Command Pattern), generate Move Right action for any character
 */
class MoveRightAction : public nextfloor::gameplay::Action {

public:
    void execute(nextfloor::character::Character* actor) final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_MOVERIGHTACTION_H_
