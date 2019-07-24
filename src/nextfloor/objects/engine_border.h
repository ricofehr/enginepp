/**
 *  @file engine_border.h
 *  @brief EngineBorder class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINEBORDER_H_
#define NEXTFLOOR_OBJECTS_ENGINEBORDER_H_

#include <glm/glm.hpp>
#include <vector>

namespace nextfloor {

namespace objects {

/**
 *  @class EngineBorder
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class EngineBorder {

public:

    EngineBorder(EngineBorder&&) = default;
    EngineBorder& operator=(EngineBorder&&) = default;
    EngineBorder(const EngineBorder&) = delete;
    EngineBorder& operator=(const EngineBorder&) = delete;
    virtual ~EngineBorder() = default;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const = 0;
    virtual void ComputeNewLocation() = 0;
    virtual bool IsObstacleInCollisionAfterPartedMove(EngineBorder* obstacle, float move_part) = 0;

    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 dimension() const = 0;
    virtual glm::vec3 move() const = 0;
    virtual float distance() const = 0;

    virtual void set_distance(float distance) = 0;
    virtual void InverseMove() = 0;

protected:

    EngineBorder() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINEBORDER_H_

