/**
 *  @file randomed_game_factory.c
 *  @brief RandomedGameFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "randomed_game_factory.h"

#include "nextfloor/gameplay/random_level.h"

namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::gameplay::Level> RandomedGameFactory::MakeLevel()
{
    using nextfloor::gameplay::RandomLevel;
    return std::make_unique<RandomLevel>();
}

}  // namespace factory

}  // namespace nextfloor