/**
 *  @file border.h
 *  @brief Border class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_BORDER_H_
#define NEXTFLOOR_MESH_BORDER_H_

#include <glm/glm.hpp>
#include <vector>

namespace nextfloor {

namespace mesh {

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
    virtual bool IsObstacleInCollisionAfterPartedMove(const Border& obstacle, float move_part) const = 0;

    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 dimension() const = 0;
    virtual glm::vec3 movement() const = 0;
    virtual float move_factor() const = 0;
    virtual bool IsMoved() const = 0;
    virtual float diagonal() const = 0;

    virtual void set_movement(const glm::vec3& movement) = 0;
    virtual void set_move_factor(float move_factor) = 0;

    virtual float CalculateWidth() const = 0;
    virtual float CalculateHeight() const = 0;
    virtual float CalculateDepth() const = 0;
    virtual glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part) const = 0;

    virtual glm::vec3 getFirstPoint() const = 0;
    virtual glm::vec3 getLastPoint() const = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_BORDER_H_
