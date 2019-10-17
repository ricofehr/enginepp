/**
 *  @file player.h
 *  @brief Player class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_PLAYER_H_
#define NEXTFLOOR_GAMEPLAY_PLAYER_H_

#include "nextfloor/gameplay/character.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/border.h"
#include "nextfloor/gameplay/camera.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class Player
 *  @brief Player model, inherits Model abstract class
 */
class Player : public Character {

public:
    Player(const glm::vec3& location, std::unique_ptr<nextfloor::objects::Border> border, std::unique_ptr<Camera> camera);
    ~Player() final = default;


    void MoveLocation() final;
    bool IsPlayer() const final { return true; }
    bool IsCamera() const final { return true; }
    Camera* camera() const final;

private:
    std::unique_ptr<Camera> camera_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_GAMEPLAY_PLAYER_H_
