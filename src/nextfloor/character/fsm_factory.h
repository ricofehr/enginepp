/**
 *  @file fsm_factory.h
 *  @brief FSMFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_FSMFACTORY_H_
#define NEXTFLOOR_CHARACTER_FSMFACTORY_H_

#include <memory>

#include "nextfloor/character/fsm.h"

namespace nextfloor {

namespace character {

/**
 *  @class FSMFactory
 *  @brief Abstract Factory Pattern for FSMs
 */
class FSMFactory {

public:
    virtual ~FSMFactory() = default;

    virtual std::unique_ptr<FSM> MakeCharacterFSM() = 0;
    virtual std::unique_ptr<State> MakeGravityCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeIdleCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeJumpUpCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeJumpDownCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeMoveDownCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeMoveLeftCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeMoveRightCharacterState(FSM* owner) const = 0;
    virtual std::unique_ptr<State> MakeMoveUpCharacterState(FSM* owner) const = 0;
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_FSMFACTORY_H_
