/**
 *  @file mesh_physic_factory.cc
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/physics/mesh_physic_factory.h"

#include "nextfloor/physics/cube_border.h"
#include "nextfloor/physics/tbb_nearer_collision_engine.h"
#include "nextfloor/physics/serial_nearer_collision_engine.h"
#include "nextfloor/physics/cl_nearer_collision_engine.h"

#include "nextfloor/polygons/mesh_polygon_factory.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace physics {

std::unique_ptr<nextfloor::objects::Border> MeshPhysicFactory::MakeBorder(const glm::vec3& location, float scale) const
{
    return MakeBorder(location, glm::vec3(scale));
}

std::unique_ptr<nextfloor::objects::Border> MeshPhysicFactory::MakeBorder(const glm::vec3& location,
                                                                          const glm::vec3& scale) const
{
    nextfloor::polygons::MeshPolygonFactory polygon_factory;
    return std::make_unique<CubeBorder>(location, scale, polygon_factory);
}

nextfloor::objects::CollisionEngine* MeshPhysicFactory::MakeCollisionEngine() const
{
    using nextfloor::core::CommonServices;

    nextfloor::objects::CollisionEngine* engine_collision{nullptr};

    /* Get parallell type from config */
    int type_parallell = CommonServices::getConfig()->getParallellAlgoType();

    switch (type_parallell) {  // clang-format off
        case NearerCollisionEngine::kPARALLELL_TBB:
            engine_collision = new TbbNearerCollisionEngine();
            break;
        case NearerCollisionEngine::kPARALLELL_CL:
            engine_collision = new ClNearerCollisionEngine();
            break;
        default:
            engine_collision = new SerialNearerCollisionEngine();
            break;
    }  // clang-format on

    assert(engine_collision != nullptr);

    return engine_collision;
}

}  // namespace physics

}  // namespace nextfloor