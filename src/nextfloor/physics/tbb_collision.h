/**
 *  @file tbb_collision.h
 *  @brief TbbCollision class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_TBBCOLLISION_H_
#define NEXTFLOOR_PHYSICS_TBBCOLLISION_H_

#include "nextfloor/physics/collision.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace physics {

/**
 *  @class TbbCollision
 *  @brief Implements inteltbbplus algorithm for collision computes
 */
class TbbCollision : public Collision {

public:

    TbbCollision();

    TbbCollision(TbbCollision&&) = default;
    TbbCollision& operator=(TbbCollision&&) = default;
    TbbCollision(const TbbCollision&) = delete;
    TbbCollision& operator=(const TbbCollision&) = delete;

    virtual ~TbbCollision() override = default;

    virtual float ComputeCollision(nextfloor::objects::EngineObject* target,
                                   nextfloor::objects::EngineObject* obstacle) override final;

private:

    virtual void InitCollisionEngine() override final;
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_TBBCOLLISION_H_
