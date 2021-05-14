/**
 *  @file jumpdown_character_state.h
 *  @brief GravityCharacterState class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_AI_GRAVITYCHARACTERSTATE_H_
#define NEXTFLOOR_AI_GRAVITYCHARACTERSTATE_H_

#include "nextfloor/ai/character_state.h"

#include "nextfloor/character/fsm.h"


namespace nextfloor {

namespace ai {

/**
 *  @class GravityCharacterState
 *  @brief Concrete class for Jump Movement State
 */
class GravityCharacterState : public CharacterState {

public:
    GravityCharacterState(nextfloor::character::FSM* owner) : owner_(owner) {}

    void Enter(nextfloor::character::Character* actor) final;

    void Execute(nextfloor::character::Character* actor, double elapsed_time) final;

    void Exit(nextfloor::character::Character* actor) final;

    bool IsInterruptible() final { return is_interruptible_; }

    int type() final { return CharacterState::kGravity; }

private:
    static constexpr float kMoveFactor = 4.0f;
    static constexpr float kDistance = 1.5f;
    static constexpr float kDuration = 0.5f;

    float current_y_stage_{0.0f};
    bool first_execution_{true};

    bool is_interruptible_{false};
    nextfloor::character::FSM* owner_{nullptr};
};

}  // namespace ai

}  // namespace nextfloor

#endif  // NEXTFLOOR_AI_JUMPCHARACTERSTATE_H_
