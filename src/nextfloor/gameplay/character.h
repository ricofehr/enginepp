/**
 *  @file character.h
 *  @brief Character class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_CHARACTER_H_
#define NEXTFLOOR_GAMEPLAY_CHARACTER_H_

#include "nextfloor/objects/model_mesh.h"

#include "nextfloor/gameplay/camera.h"

namespace nextfloor {

namespace gameplay {

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
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_CHARACTER_H_
