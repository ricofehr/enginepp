/**
 *  @file character.h
 *  @brief Character class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_CHARACTER_H_
#define NEXTFLOOR_CHARACTER_CHARACTER_H_

#include "nextfloor/objects/model_mesh.h"

#include "nextfloor/character/camera.h"

namespace nextfloor {

namespace character {

/**
 *  @class Character
 *  @brief Abstract class who defines moving character
 */
class Character : public nextfloor::objects::ModelMesh {

public:
    ~Character() override = default;

    virtual Camera* camera() const = 0;

protected:
    Character() = default;

    Character(Character&&) = delete;
    Character& operator=(Character&&) = delete;
    Character(const Character&) = delete;
    Character& operator=(const Character&) = delete;

    void reset_movement() { set_movement(glm::vec3(kNoMovement)); }

private:
    static constexpr float kNoMovement = 0.0f;
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_CHARACTER_H_
