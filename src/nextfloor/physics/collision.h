/**
 *  @file collision.h
 *  @brief Collision class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_COLLISION_H_
#define NEXTFLOOR_PHYSICS_COLLISION_H_

#include <tbb/mutex.h>

#include "nextfloor/objects/engine_collision.h"

namespace nextfloor {

namespace physics {

using nextfloor::objects::EngineCollision;

/**
 *  @class Collision
 *  @brief Interface who manage collisition computes between 3d models\n
 *  Use Strategy / Template Method Patterns for this abstract class and subclasses,\n
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus vernsion and opencl version
 */
class Collision : public EngineCollision {

public:

    static constexpr int kPARALLELL_SERIAL = 1;
    static constexpr int kPARALLELL_TBB = 2;
    static constexpr int kPARALLELL_CL = 3;

    Collision(Collision&&) = default;

    Collision& operator=(Collision&&) = default;

    Collision(const Collision&) = delete;

    Collision& operator=(const Collision&) = delete;

    /* Abstract Class, define virtual destructor */
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

    /* Default Construtor : Protected scope ensures Abstract Class Design */
    Collision() = default;

    virtual void InitCollisionEngine() = 0;

    /* Accuracy (computes count) of Collision detection */
    int granularity_{16};
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_COLLISION_H_
