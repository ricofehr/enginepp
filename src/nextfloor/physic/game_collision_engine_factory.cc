/**
 *  @file game_collision_engine_factory.cc
 *  @brief GameCollisionEngineFactory Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/physic/game_collision_engine_factory.h"

#include "nextfloor/physic/tbb_nearer_collision_engine.h"
#include "nextfloor/physic/serial_nearer_collision_engine.h"
#include "nextfloor/physic/cl_nearer_collision_engine.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace physic {

std::unique_ptr<CollisionEngine> GameCollisionEngineFactory::MakeCollisionEngine(int type, int granularity) const
{
    std::unique_ptr<CollisionEngine> engine_collision{nullptr};

    switch (type) {  // clang-format off
        case NearerCollisionEngine::kPARALLELL_TBB:
            engine_collision = std::make_unique<TbbNearerCollisionEngine>(granularity);
            break;
        case NearerCollisionEngine::kPARALLELL_CL:
            engine_collision = std::make_unique<ClNearerCollisionEngine>(granularity);
            break;
        default:
            engine_collision = std::make_unique<SerialNearerCollisionEngine>(granularity);
            break;
    }  // clang-format on

    assert(engine_collision != nullptr);

    return engine_collision;
}

}  // namespace physic

}  // namespace nextfloor
