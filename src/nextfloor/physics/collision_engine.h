/**
 *  @file collision_engine.h
 *  @brief CollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_COLLISIONNEXTFLOOR_H_
#define PHYSICS_COLLISIONNEXTFLOOR_H_

#include <tbb/mutex.h>

namespace nextfloor {

namespace universe {
    class Model3D;
}

namespace physics {

/**
 *  @class CollisionEngine
 *  @brief Interface who manage collisition computes between 3d models\n
 *  Use Strategy / Template Method Patterns for this abstract class and subclasses,\n
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus vernsion and opencl version
 */
class CollisionEngine {

public:

    static constexpr int kPARALLELL_SERIAL = 1;
    static constexpr int kPARALLELL_TBB = 2;
    static constexpr int kPARALLELL_CL = 3;

    CollisionEngine(CollisionEngine&&) = default;

    CollisionEngine& operator=(CollisionEngine&&) = default;

    CollisionEngine(const CollisionEngine&) = delete;

    CollisionEngine& operator=(const CollisionEngine&) = delete;

    /* Abstract Class, define virtual destructor */
    virtual ~CollisionEngine() = default;

    /* Template Method : Detect if a collision exists between target and obstacle. */
    virtual void DetectCollision(nextfloor::universe::Model3D* target,
                                 nextfloor::universe::Model3D* obstacle);

    /**
     *  Primitive Operation subclassed: compute collision distance between borders of 2 objects
     *  @return parted (as fraction of setted move) distance between the 2 borders
     */
    virtual float ComputeCollision(nextfloor::universe::Model3D* target,
                                   nextfloor::universe::Model3D* obstacle) = 0;

protected:

    /* Default Construtor : Protected scope ensures Abstract Class Design */
    CollisionEngine() = default;

    virtual void InitCollisionEngine() = 0;

    /* Accuracy (computes count) of Collision detection */
    int granularity_{16};
};

} // namespace physics

} // namespace nextfloor

#endif // PHYSICS_COLLISIONNEXTFLOOR_H_
