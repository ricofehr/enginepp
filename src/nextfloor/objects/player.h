/**
 *  @file player.h
 *  @brief Player class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_PLAYER_H_
#define NEXTFLOOR_OBJECTS_PLAYER_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>


namespace nextfloor {

namespace objects {

/**
 *  @class Player
 *  @brief Player model, inherits Model abstract class
 */
class Player : public ModelMesh {

public:
    Player(const glm::vec3& location);
    ~Player() final = default;


    void Move() final;
    bool IsPlayer() const final { return true; }
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_PLAYER_H_