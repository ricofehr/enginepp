/**
 *  @file character_fsm.cc
 *  @brief CharacterFSM class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/character_fsm.h"

#include <iostream>


namespace nextfloor {

namespace ai {

CharacterFSM::CharacterFSM(nextfloor::character::FSMFactory* fsm_factory)
{
    fsm_factory_ = fsm_factory;
    current_state_ = fsm_factory_->MakeIdleCharacterState(this);
    previous_state_ = fsm_factory_->MakeIdleCharacterState(this);
}

void CharacterFSM::JumpUp()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeJumpUpCharacterState(this));
    }
}

void CharacterFSM::JumpDown()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeJumpDownCharacterState(this));
    }
}

void CharacterFSM::MoveUp()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeMoveUpCharacterState(this));
    }
}

void CharacterFSM::MoveDown()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeMoveDownCharacterState(this));
    }
}

void CharacterFSM::MoveLeft()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeMoveLeftCharacterState(this));
    }
}

void CharacterFSM::MoveRight()
{
    assert(fsm_factory_ != nullptr);

    if (IsInterruptible()) {
        change_state(fsm_factory_->MakeMoveRightCharacterState(this));
    }
}

void CharacterFSM::Idle()
{
    assert(fsm_factory_ != nullptr);
    assert(current_state_ != nullptr);

    if (IsInterruptible()) {
        if (current_state_->type() != CharacterState::kIdle) {
            change_state(fsm_factory_->MakeIdleCharacterState(this));
        }
    }
}


void CharacterFSM::Update(double elapsed_time)
{
    assert(current_state_ != nullptr);
    assert(owner_ != nullptr);

    if (current_state_->IsFinished()) {
        current_state_->Exit(owner_);
    }

    current_state_->Execute(owner_, elapsed_time);
}

void CharacterFSM::change_state(std::unique_ptr<nextfloor::character::State> new_state)
{
    assert(current_state_ != nullptr);
    assert(previous_state_ != nullptr);
    assert(owner_ != nullptr);

    if (IsInterruptible()) {
        previous_state_.swap(current_state_);
        current_state_.swap(new_state);
        current_state_->Enter(owner_);
        new_state.reset();
    }
}

void CharacterFSM::revert_to_previous_state()
{
    assert(current_state_ != nullptr);
    assert(previous_state_ != nullptr);

    if (IsInterruptible()) {
        current_state_->Exit(owner_);
        current_state_.swap(previous_state_);
        current_state_->Enter(owner_);
    }
}

bool CharacterFSM::is_in_state(int state_type) const
{
    assert(current_state_ != nullptr);

    return (current_state_->type() == state_type);
}

bool CharacterFSM::IsInterruptible() const
{
    assert(current_state_ != nullptr);
    assert(owner_ != nullptr);

    return current_state_->IsInterruptible();
}

}  // namespace ai

}  // namespace nextfloor