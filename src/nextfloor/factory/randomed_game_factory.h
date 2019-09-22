/**
 *  @file randomed_game_factory.h
 *  @brief RandomedGameFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RANDOMED_GAME_FACTORY_H_
#define NEXTFLOOR_RANDOMED_GAME_FACTORY_H_

#include "nextfloor/factory/game_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class RandomedGameFactory
 *  @brief RandomedGameFactory make Universe objects in random manner
 */
class RandomedGameFactory : public GameFactory {

public:
    std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const final;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_RANDOMED_GAME_FACTORY_H_
