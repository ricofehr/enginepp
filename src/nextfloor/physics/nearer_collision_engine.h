/**
 *  @file nearer_collision_engine.h
 *  @brief NearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_NEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSICS_NEARERCOLLISIONENGINE_H_

#include "nextfloor/objects/collision_engine.h"

#include <tbb/mutex.h>

#include "nextfloor/objects/mesh.h"


namespace nextfloor {

namespace physics {

/**
 *  @class NearerCollisionEngine
 *  @brief Abstract Class who manage collisition computes between 3d models\n
 *  Use Strategy / Template Method Patterns for this abstract class and subclasses,\n
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus version and opencl version
 */
class NearerCollisionEngine : public nextfloor::objects::CollisionEngine {

public:

    static constexpr int kPARALLELL_SERIAL = 1;
    static constexpr int kPARALLELL_TBB = 2;
    static constexpr int kPARALLELL_CL = 3;

    virtual ~NearerCollisionEngine() = default;

    /* Template Method : Detect if a collision exists between target and obstacle. */
    virtual void DetectCollision(nextfloor::objects::Mesh* target,
                                 nextfloor::objects::Mesh* obstacle);

    /**
     *  Primitive Operation subclassed: compute collision distance between borders of 2 objects
     *  @return parted (as fraction of setted move) distance between the 2 borders
     */
    virtual float ComputeCollision(nextfloor::objects::Mesh* target,
                                   nextfloor::objects::Mesh* obstacle) = 0;

protected:

    NearerCollisionEngine() = default;

    NearerCollisionEngine(NearerCollisionEngine&&) = default;
    NearerCollisionEngine& operator=(NearerCollisionEngine&&) = default;
    NearerCollisionEngine(const NearerCollisionEngine&) = delete;
    NearerCollisionEngine& operator=(const NearerCollisionEngine&) = delete;

    virtual void InitCollisionEngine() = 0;

    /* Accuracy (computes count) of NearerCollisionEngine detection */
    int granularity_{16};
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_NEARERCOLLISIONENGINE_H_
