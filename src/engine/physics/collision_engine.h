/*
 * CollisionEngine class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Use Strategy Pattern for this abstract class and subclasses,
 * which proposes 3 differents Collision Algorithm: serial, cilkplus vernsion and opencl version
 */


#ifndef PHYSICS_COLLISIONENGINE_H_
#define PHYSICS_COLLISIONENGINE_H_

#include <tbb/mutex.h>

namespace engine {

namespace universe {
    /* Forward declaration of Model3D class */
    class Model3D;
} //namespace universe


namespace physics {

class CollisionEngine {

public:
    static const int kPARALLELL_SERIAL = 1;
    static const int kPARALLELL_CILK = 2;
    static const int kPARALLELL_CL = 3;

    virtual float ComputeCollision(float box1[], float box2[]) = 0;
    virtual std::vector<engine::universe::Model3D*> DetectCollision(engine::universe::Model3D *object,
                                                                    engine::universe::Model3D *obstacle,
                                                                    tbb::mutex &collision_mutex);
    virtual ~CollisionEngine() = default;

protected:
    virtual void InitCollisionEngine() = 0;

    CollisionEngine() = default;
    CollisionEngine(const CollisionEngine&) = default;
    CollisionEngine& operator=(const CollisionEngine&) = default;

    int granularity_{16};
};

}//namespace physics
}//namespace engine

#endif //PHYSICS_COLLISIONENGINE_H_
