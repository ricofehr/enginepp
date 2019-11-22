/**
 *  @file game_character_factory.cc
 *  @brief GameCharacterFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/character/game_character_factory.h"

#include "nextfloor/character/player.h"

namespace nextfloor {

namespace character {

GameCharacterFactory::GameCharacterFactory(CameraFactory* camera_factory, nextfloor::mesh::PhysicFactory* physic_factory)
{
    camera_factory_ = camera_factory;
    physic_factory_ = physic_factory;
}

std::unique_ptr<Character> GameCharacterFactory::MakePlayer(const glm::vec3& location) const
{
    assert(physic_factory_ != nullptr);
    assert(camera_factory_ != nullptr);

    auto border = physic_factory_->MakeBorder(location, Player::kBorderScale);
    auto camera = camera_factory_->MakeCamera();
    return std::make_unique<Player>(std::move(border), std::move(camera));
}

}  // namespace character

}  // namespace nextfloor