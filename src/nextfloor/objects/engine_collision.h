/**
 *  @file engine_collision.h
 *  @brief EngineCollision class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINECOLLISION_H_
#define NEXTFLOOR_OBJECTS_ENGINECOLLISION_H_

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace objects {

/**
 *  @class EngineCollision
 *  @brief Interface who manage collisition computes between 3d models\n
 */
class EngineCollision {

public:

    EngineCollision(EngineCollision&&) = default;
    EngineCollision& operator=(EngineCollision&&) = default;
    EngineCollision(const EngineCollision&) = delete;
    EngineCollision& operator=(const EngineCollision&) = delete;
    virtual ~EngineCollision() = default;

    /* Template Method : Detect if a collision exists between target and obstacle. */
    virtual void DetectCollision(EngineObject* target,
                                 EngineObject* obstacle) = 0;

    /**
     *  Primitive Operation subclassed: compute collision distance between borders of 2 objects
     *  @return parted (as fraction of setted move) distance between the 2 borders
     */
    virtual float ComputeCollision(EngineObject* target,
                                   EngineObject* obstacle) = 0;

protected:

    EngineCollision() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINECOLLISION_H_
