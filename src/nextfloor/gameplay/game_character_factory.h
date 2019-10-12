/**
 *  @file game_character_factory.h
 *  @brief Concrete Factory Class Implementation for player part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_GAMECHARACTERFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_GAMECHARACTERFACTORY_H_

#include "nextfloor/objects/character_factory.h"

#include "nextfloor/objects/physic_factory.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class GameCharacterFactory
 *  @brief Concrete Factory for player related objects
 */
class GameCharacterFactory : public nextfloor::objects::CharacterFactory {

public:
    GameCharacterFactory(nextfloor::objects::PhysicFactory* physic_factory);
    ~GameCharacterFactory() final = default;

    std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(const glm::vec3& location) const override;
    std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner = nullptr) const override;

private:
    nextfloor::objects::PhysicFactory* physic_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_GAMECHARACERFACTORY_H_
