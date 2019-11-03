/**
 *  @file game_character_factory.h
 *  @brief GameCharacterFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_GAMECHARACTERFACTORY_H_
#define NEXTFLOOR_CHARACTER_GAMECHARACTERFACTORY_H_

#include "nextfloor/character/character_factory.h"

#include "nextfloor/character/camera_factory.h"
#include "nextfloor/objects/physic_factory.h"

namespace nextfloor {

namespace character {

/**
 *  @class GameCharacterFactory
 *  @brief Concrete Factory for game character related objects
 */
class GameCharacterFactory : public CharacterFactory {

public:
    GameCharacterFactory(CameraFactory* camera_factory, nextfloor::objects::PhysicFactory* physic_factory);
    ~GameCharacterFactory() final = default;

    std::unique_ptr<Character> MakePlayer(const glm::vec3& location) const override;

private:
    CameraFactory* camera_factory_{nullptr};
    nextfloor::objects::PhysicFactory* physic_factory_{nullptr};
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_GAMECHARACERFACTORY_H_
