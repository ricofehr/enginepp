/*
 *  CollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Interface who manage collisition computes between 3d models
 *  Use Strategy Pattern for this abstract class and subclasses,
 *  which proposes 3 differents Collision Algorithm: serial, cilkplus vernsion and opencl version
 */


#ifndef PHYSICS_COLLISIONENGINE_H_
#define PHYSICS_COLLISIONENGINE_H_

#include <tbb/mutex.h>

namespace engine {

/*
 *  Forward declaration of 
 *  Model3D class
 */
namespace universe {
    class Model3D;
} // namespace universe

namespace physics {

class CollisionEngine {

public:

    /*
     *  Parallell Algo Constants
     */
    static const int kPARALLELL_SERIAL = 1;
    static const int kPARALLELL_CILK = 2;
    static const int kPARALLELL_CL = 3;

    /*
     *  Detect if a collision exists between object and obstacle.
     *  collision_mutex is used by ensure thread safe compute.
     *  Return a list of Model3D to recompute if there was other crossed objects before.
     */
    virtual std::vector<engine::universe::Model3D*> DetectCollision(engine::universe::Model3D* object,
                                                                    engine::universe::Model3D* obstacle);
    /*
     *  Compute the distance of collision between 2 borders box (box1 and box2 are Coordinates arrays)
     *  Return the distance (-1.0f if no collision point)
     */
    virtual float ComputeCollision(float box1[], float box2[]) = 0;

protected:

    /*
     *  Construtor
     *  Protected scope ensures Interface Design
     */
    CollisionEngine() = default;
    CollisionEngine(const CollisionEngine&) = default;
    CollisionEngine& operator=(const CollisionEngine&) = default;

    /*
     *  Default Destructor
     *  Is Virtual because Interface Class
     */
    virtual ~CollisionEngine() = default;

    /*
     *  Init the context of Collision Engine
     */
    virtual void InitCollisionEngine() = 0;

    /*
     *  Define accuracy of Collision computes
     */
    int granularity_{16};
};

} // namespace physics

} // namespace engine

#endif // PHYSICS_COLLISIONENGINE_H_
