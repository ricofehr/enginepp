/**
 *  @file character_fsm_factory.cc
 *  @brief CharacterFSMFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/ai/character_fsm_factory.h"

#include "nextfloor/ai/character_fsm.h"
#include "nextfloor/ai/gravity_character_state.h"
#include "nextfloor/ai/idle_character_state.h"
#include "nextfloor/ai/jumpup_character_state.h"
#include "nextfloor/ai/jumpdown_character_state.h"
#include "nextfloor/ai/movedown_character_state.h"
#include "nextfloor/ai/moveleft_character_state.h"
#include "nextfloor/ai/moveright_character_state.h"
#include "nextfloor/ai/moveup_character_state.h"

namespace nextfloor {

namespace ai {


std::unique_ptr<nextfloor::character::FSM> CharacterFSMFactory::MakeCharacterFSM()
{
    return std::make_unique<CharacterFSM>(this);
}

std::unique_ptr<nextfloor::character::State> CharacterFSMFactory::MakeGravityCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<GravityCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State> CharacterFSMFactory::MakeIdleCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<IdleCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeJumpUpCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<JumpUpCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeJumpDownCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<JumpDownCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeMoveDownCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<MoveDownCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeMoveLeftCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<MoveLeftCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeMoveRightCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<MoveRightCharacterState>(owner);
}

std::unique_ptr<nextfloor::character::State>
  CharacterFSMFactory::MakeMoveUpCharacterState(nextfloor::character::FSM* owner) const
{
    return std::make_unique<MoveUpCharacterState>(owner);
}

}  // namespace ai

}  // namespace nextfloor
