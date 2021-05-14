/**
 *  @file character_factory.h
 *  @brief CharacterFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_CHARACTERFACTORY_H_
#define NEXTFLOOR_CHARACTER_CHARACTERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/character/character.h"

namespace nextfloor {

namespace character {

/**
 *  @class CharacterFactory
 *  @brief Abstract Factory Pattern for Characters
 */
class CharacterFactory {

public:
    virtual ~CharacterFactory() = default;
    virtual std::unique_ptr<Character> MakePlayer(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Character> MakeMovingRock(const glm::vec3& location, const glm::vec3& movement) const = 0;
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_CHARACTERFACTORY_H_
