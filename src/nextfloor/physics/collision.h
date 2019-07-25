/**
 *  @file collision.h
 *  @brief Collision class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_COLLISION_H_
#define NEXTFLOOR_PHYSICS_COLLISION_H_

#include "nextfloor/objects/engine_collision.h"

#include <tbb/mutex.h>

#include "nextfloor/objects/engine_object.h"


namespace nextfloor {

namespace physics {

/**
 *  @class Collision
 *  @brief Abstract Class who manage collisition computes between 3d models\n
 *  Use Strategy / Template Method Patterns for this abstract class and subclasses,\n
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus version and opencl version
 */
class Collision : public nextfloor::objects::EngineCollision {

public:

    static constexpr int kPARALLELL_SERIAL = 1;
    static constexpr int kPARALLELL_TBB = 2;
    static constexpr int kPARALLELL_CL = 3;

    virtual ~Collision() = default;

    /* Template Method : Detect if a collision exists between target and obstacle. */
    virtual void DetectCollision(nextfloor::objects::EngineObject* target,
                                 nextfloor::objects::EngineObject* obstacle);

    /**
     *  Primitive Operation subclassed: compute collision distance between borders of 2 objects
     *  @return parted (as fraction of setted move) distance between the 2 borders
     */
    virtual float ComputeCollision(nextfloor::objects::EngineObject* target,
                                   nextfloor::objects::EngineObject* obstacle) = 0;

protected:

    Collision() = default;

    Collision(Collision&&) = default;
    Collision& operator=(Collision&&) = default;
    Collision(const Collision&) = delete;
    Collision& operator=(const Collision&) = delete;

    virtual void InitCollisionEngine() = 0;

    /* Accuracy (computes count) of Collision detection */
    int granularity_{16};
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_COLLISION_H_
