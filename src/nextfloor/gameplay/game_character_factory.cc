/**
 *  @file game_character_factory.cc
 *  @brief Factory Class for characters
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/gameplay/game_character_factory.h"

#include "nextfloor/gameplay/head_camera.h"
#include "nextfloor/gameplay/player.h"

namespace nextfloor {

namespace gameplay {

GameCharacterFactory::GameCharacterFactory(nextfloor::objects::PhysicFactory* physic_factory)
{
    physic_factory_ = physic_factory;
}

std::unique_ptr<Character> GameCharacterFactory::MakePlayer(const glm::vec3& location) const
{
    auto border = physic_factory_->MakeBorder(location, glm::vec3(0.4f));
    auto camera = MakeCamera();
    return std::make_unique<Player>(location, std::move(border), std::move(camera));
}

std::unique_ptr<Camera> GameCharacterFactory::MakeCamera() const
{
    return std::make_unique<HeadCamera>(3.14f, 0.0f);
}


}  // namespace gameplay

}  // namespace nextfloor