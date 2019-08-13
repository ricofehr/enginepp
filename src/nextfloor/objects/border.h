/**
 *  @file border.h
 *  @brief Border class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_BORDER_H_
#define NEXTFLOOR_OBJECTS_BORDER_H_

#include <glm/glm.hpp>
#include <vector>

namespace nextfloor {

namespace objects {

/**
 *  @class Border
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class Border {

public:

    virtual ~Border() = default;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const = 0;
    virtual void ComputeNewLocation() = 0;
    virtual bool IsObstacleInCollisionAfterPartedMove(Border* obstacle, float move_part) = 0;

    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 dimension() const = 0;
    virtual glm::vec3 movement() const = 0;
    virtual float move_factor() const = 0;
    virtual bool IsMoved() const noexcept = 0;
    virtual float diagonal() const noexcept = 0;

    virtual void set_movement(glm::vec3 movement) const = 0;
    virtual void set_move_factor(float move_factor) const = 0;

    virtual float CalculateWidth() = 0;
    virtual float CalculateHeight() = 0;
    virtual float CalculateDepth() = 0;
    virtual glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part) = 0;

protected:

    Border() = default;

    Border(Border&&) = default;
    Border& operator=(Border&&) = default;
    Border(const Border&) = delete;
    Border& operator=(const Border&) = delete;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_BORDER_H_

