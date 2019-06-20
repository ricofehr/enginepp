/**
 *  @file tbb_collision_engine.h
 *  @brief TbbCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_TBBCOLLISIONNEXTFLOOR_H_
#define PHYSICS_TBBCOLLISIONNEXTFLOOR_H_

#include <tbb/mutex.h>

#include "nextfloor/physics/collision_engine.h"

namespace nextfloor {

namespace physics {

/**
 *  @class TbbCollisionEngine
 *  @brief Implements inteltbbplus algorithm for collision computes
 */
class TbbCollisionEngine : public CollisionEngine {

public:

    TbbCollisionEngine(TbbCollisionEngine&&) = default;

    TbbCollisionEngine& operator=(TbbCollisionEngine&&) = default;

    TbbCollisionEngine(const TbbCollisionEngine&) = delete;

    TbbCollisionEngine& operator=(const TbbCollisionEngine&) = delete;

    ~TbbCollisionEngine() override = default;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
     */
    inline static TbbCollisionEngine* Instance()
    {
        static auto sInstance = new TbbCollisionEngine;
        return sInstance;
    }

    virtual float ComputeCollision(nextfloor::universe::Model3D* target,
                                   nextfloor::universe::Model3D* obstacle) override final;

private:

    TbbCollisionEngine();

    virtual void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace nextfloor

#endif // PHYSICS_TBBCOLLISIONNEXTFLOOR_H_
