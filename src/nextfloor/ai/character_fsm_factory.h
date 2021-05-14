/**
 *  @file character_fsm_factory.h
 *  @brief CharacterFSMFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_AI_CHARACTERFSMFACTORY_H_
#define NEXTFLOOR_AI_CHARACTERFSMFACTORY_H_

#include "nextfloor/character/fsm_factory.h"


namespace nextfloor {

namespace ai {

/**
 *  @class CharacterFSMFactory
 *  @brief Concrete Factory for FSM character related objects
 */
class CharacterFSMFactory : public nextfloor::character::FSMFactory {

public:
    std::unique_ptr<nextfloor::character::FSM> MakeCharacterFSM() override;
    std::unique_ptr<nextfloor::character::State> MakeGravityCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeIdleCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeJumpUpCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeJumpDownCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeMoveDownCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeMoveLeftCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeMoveRightCharacterState(nextfloor::character::FSM* owner) const override;
    std::unique_ptr<nextfloor::character::State> MakeMoveUpCharacterState(nextfloor::character::FSM* owner) const override;
};

}  // namespace ai

}  // namespace nextfloor

#endif  // NEXTFLOOR_AI_CHARACTERFSMFACTORY_H_
