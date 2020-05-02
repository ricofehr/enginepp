/**
 *  @file character_state.h
 *  @brief CharacterState class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_AI_CHARACTERSTATE_H_
#define NEXTFLOOR_AI_CHARACTERSTATE_H_

#include "nextfloor/character/state.h"

#include "nextfloor/character/character.h"

namespace nextfloor {

namespace ai {

/**
 *  @class CharacterState
 *  @brief Abstract class who implement State node in a State Pattern
 */
class CharacterState : public nextfloor::character::State {


public:
    static constexpr int kIdle = 0;
    static constexpr int kMoved = 1;

    static constexpr int kJumpUp = 2;
    static constexpr int kJumpDown = 3;

    ~CharacterState() override = default;

    bool IsFinished() override { return is_finished_; }

protected:
    CharacterState() = default;

    CharacterState(CharacterState&&) = delete;
    CharacterState& operator=(CharacterState&&) = delete;
    CharacterState(const CharacterState&) = delete;
    CharacterState& operator=(const CharacterState&) = delete;

    bool is_finished_ = false;
};

}  // namespace ai

}  // namespace nextfloor

#endif  // NEXTFLOOR_AI_CHARACTERSTATE_H_
