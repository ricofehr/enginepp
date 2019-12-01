/**
 *  @file game_collision_engine_factory.h
 *  @brief GameCollisionEngineFactory Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_GAMECOLLISIONENGINEFACTORY_H_
#define NEXTFLOOR_PHYSIC_GAMECOLLISIONENGINEFACTORY_H_

#include "nextfloor/physic/collision_engine_factory.h"

namespace nextfloor {

namespace physic {

/**
 *  @class CollisionEngineFactory
 *  @brief Concrete factory for Collision Engines
 */
class GameCollisionEngineFactory : public CollisionEngineFactory {

public:
    std::unique_ptr<CollisionEngine> MakeCollisionEngine(int type, int granularity) const final;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_GAMECOLLISIONENGINEFACTORY_H_
