/**
 *  @file character_fsm.h
 *  @brief CharacterFSM class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_AI_CHARACTERFSM_H_
#define NEXTFLOOR_AI_CHARACTERFSM_H_

#include "nextfloor/character/fsm.h"

#include <memory>

#include "nextfloor/ai/character_state.h"
#include "nextfloor/character/state.h"
#include "nextfloor/character/fsm_factory.h"

namespace nextfloor {

namespace ai {

/**
 *  @class CharacterFSM
 *  @brief Abstract class who implement FSM Pattern
 */
class CharacterFSM : public nextfloor::character::FSM {

public:
    CharacterFSM(nextfloor::character::FSMFactory* fsm_factory);

    void JumpUp() override;
    void JumpDown() override;
    void MoveUp() override;
    void MoveDown() override;
    void MoveLeft() override;
    void MoveRight() override;
    void Idle() override;
    void Update(double elapsed_time) override;

    void set_owner(nextfloor::character::Character* owner) override { owner_ = owner; }

protected:
    void change_state(std::unique_ptr<nextfloor::character::State> new_state);
    void revert_to_previous_state();
    bool is_in_state(int state_type) const;
    bool IsInterruptible() const;

    nextfloor::character::Character* owner_{nullptr};
    std::unique_ptr<nextfloor::character::State> current_state_{nullptr};
    std::unique_ptr<nextfloor::character::State> previous_state_{nullptr};
    nextfloor::character::FSMFactory* fsm_factory_{nullptr};
};

}  // namespace ai

}  // namespace nextfloor

#endif  // NEXTFLOOR_AI_CHARACTERFSM_H_
