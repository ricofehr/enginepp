/**
 *  @file collision_engine.h
 *  @brief CollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_COLLISIONNEXTFLOOR_H_
#define PHYSICS_COLLISIONNEXTFLOOR_H_

#include <tbb/mutex.h>

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/*
 *  Forward declaration of
 *  Model3D class
 */
namespace universe {
    class Model3D;
} // namespace universe

/**
 *  @namespace nextfloor::physics
 *  @brief Physics and collision engines
 */
namespace physics {

/**
 *  @class CollisionEngine
 *  @brief Interface who manage collisition computes between 3d models\n
 *  Use Strategy Pattern for this abstract class and subclasses,\n
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus vernsion and opencl version
 */
class CollisionEngine {

public:

    /**
     *  Default Move constructor
     */
    CollisionEngine(CollisionEngine&&) = default;

    /**
     *  Default Move assignment
     */
    CollisionEngine& operator=(CollisionEngine&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    CollisionEngine(const CollisionEngine&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    CollisionEngine& operator=(const CollisionEngine&) = delete;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~CollisionEngine() = default;

    /**
     *  Parallell Algo Constants
     */
    static constexpr int kPARALLELL_SERIAL = 1;
    static constexpr int kPARALLELL_TBB = 2;
    static constexpr int kPARALLELL_CL = 3;

    /**
     *  Detect if a collision exists between target and obstacle.
     *  @param target: the first object, targetted for the detection
     *  @param obstacle: the second object used for the detection
     */
    virtual void DetectCollision(nextfloor::universe::Model3D* target,
                                 nextfloor::universe::Model3D* obstacle);

    /**
     *  Compute collision distance between borders of 2 objects
     *  A pure virtual member.
     *  @param box1 includes the coords for the first border and the moving vector
     *  @param box2 includes the coords for the second border and the moving vector
     *  @return distance between the 2 borders
     */
    virtual float ComputeCollision(float box1[], float box2[]) = 0;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    CollisionEngine() = default;

    /**
     *  Init the context of Collision Engine
     *  A pure virtual member.
     */
    virtual void InitCollisionEngine() = 0;


    /** Accuracy (computes count) of Collision detection */
    int granularity_{16};
};

} // namespace physics

} // namespace nextfloor

#endif // PHYSICS_COLLISIONNEXTFLOOR_H_
