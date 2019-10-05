/**
 *  @file player.h
 *  @brief Player class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_PLAYER_H_
#define NEXTFLOOR_GAMEPLAY_PLAYER_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>


namespace nextfloor {

namespace gameplay {

/**
 *  @class Player
 *  @brief Player model, inherits Model abstract class
 */
class Player : public nextfloor::objects::ModelMesh {

public:
    Player(const glm::vec3& location);
    ~Player() final = default;


    void Move() final;
    bool IsPlayer() const final { return true; }
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_GAMEPLAY_PLAYER_H_