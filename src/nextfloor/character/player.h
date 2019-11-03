/**
 *  @file player.h
 *  @brief Player class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_PLAYER_H_
#define NEXTFLOOR_CHARACTER_PLAYER_H_

#include "nextfloor/character/character.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace character {

/**
 *  @class Player
 *  @brief Game Player implementation
 */
class Player : public Character {

public:
    static constexpr float kBorderScale = 0.4f;

    Player(std::unique_ptr<nextfloor::objects::Border> border, std::unique_ptr<Camera> camera);
    ~Player() final = default;

    void MoveLocation() final;
    bool IsPlayer() const final { return true; }
    bool IsCamera() const final { return true; }
    Camera* camera() const final;

private:
    std::unique_ptr<Camera> camera_{nullptr};
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_PLAYER_H_
