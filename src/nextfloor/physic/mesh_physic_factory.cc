/**
 *  @file mesh_physic_factory.cc
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/physic/mesh_physic_factory.h"

#include "nextfloor/physic/cube_border.h"
#include "nextfloor/physic/tbb_nearer_collision_engine.h"
#include "nextfloor/physic/serial_nearer_collision_engine.h"
#include "nextfloor/physic/cl_nearer_collision_engine.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace physic {

std::unique_ptr<nextfloor::mesh::Border> MeshPhysicFactory::MakeBorder(const glm::vec3& location, float scale) const
{
    return MakeBorder(location, glm::vec3(scale));
}

std::unique_ptr<nextfloor::mesh::Border> MeshPhysicFactory::MakeBorder(const glm::vec3& location,
                                                                       const glm::vec3& scale) const
{
    return std::make_unique<CubeBorder>(location, scale);
}

std::unique_ptr<nextfloor::gameplay::CollisionEngine> MeshPhysicFactory::MakeCollisionEngine() const
{
    using nextfloor::core::CommonServices;

    std::unique_ptr<nextfloor::gameplay::CollisionEngine> engine_collision{nullptr};

    /* Get parallell type from config */
    int type_parallell = CommonServices::getConfig()->getParallellAlgoType();

    switch (type_parallell) {  // clang-format off
        case NearerCollisionEngine::kPARALLELL_TBB:
            engine_collision = std::make_unique<TbbNearerCollisionEngine>();
            break;
        case NearerCollisionEngine::kPARALLELL_CL:
            engine_collision = std::make_unique<ClNearerCollisionEngine>();
            break;
        default:
            engine_collision = std::make_unique<SerialNearerCollisionEngine>();
            break;
    }  // clang-format on

    assert(engine_collision != nullptr);

    return engine_collision;
}

}  // namespace physic

}  // namespace nextfloor
