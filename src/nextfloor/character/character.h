/**
 *  @file character.h
 *  @brief Character class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_CHARACTER_H_
#define NEXTFLOOR_CHARACTER_CHARACTER_H_

#include "nextfloor/mesh/dynamic_mesh.h"

#include <glm/glm.hpp>

#include "nextfloor/character/camera.h"

namespace nextfloor {

namespace character {

/**
 *  @class Character
 *  @brief Abstract class who defines moving character
 */
class Character : public nextfloor::mesh::DynamicMesh {

public:
    ~Character() override = default;

    virtual Camera* camera() const = 0;

    std::string class_name() final { return "Character"; }

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
