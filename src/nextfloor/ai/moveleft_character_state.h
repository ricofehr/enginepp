/**
 *  @file moveleft_character_state.h
 *  @brief MoveLeftCharacterState class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_AI_MOVELEFTCHARACTERSTATE_H_
#define NEXTFLOOR_AI_MOVELEFTCHARACTERSTATE_H_

#include "nextfloor/ai/character_state.h"

#include "nextfloor/character/fsm.h"


namespace nextfloor {

namespace ai {

/**
 *  @class MoveLeftCharacterState
 *  @brief Concrete class for Move Movement State
 */
class MoveLeftCharacterState : public CharacterState {

public:
    MoveLeftCharacterState(nextfloor::character::FSM* owner) : owner_(owner) {}

    void Enter(nextfloor::character::Character* actor) final;

    void Execute(nextfloor::character::Character* actor, double elapsed_time) final;

    void Exit(nextfloor::character::Character* actor) final;

    bool IsInterruptible() final { return true; }

    int type() final { return CharacterState::kMoved; }

private:
    static constexpr float kMoveFactor = 2.0f;

    nextfloor::character::FSM* owner_{nullptr};
};

}  // namespace ai

}  // namespace nextfloor

#endif  // NEXTFLOOR_AI_MOVELEFTCHARACTERSTATE_H_
