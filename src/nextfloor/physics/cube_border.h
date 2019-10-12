/**
 *  @file cube_border.h
 *  @brief CubeBorder class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_CUBEBORDER_H_
#define NEXTFLOOR_PHYSICS_CUBEBORDER_H_

#include "nextfloor/objects/border.h"

//#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <vector>
#include <memory>

#include "nextfloor/polygons/cube.h"
#include "nextfloor/objects/polygon_factory.h"

namespace nextfloor {

namespace physics {

/**
 *  @class CubeBorder
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class CubeBorder : public nextfloor::objects::Border {

public:
    CubeBorder(const glm::vec3& location, const glm::vec3& scale, const nextfloor::objects::PolygonFactory& factory);
    CubeBorder(const glm::vec3& location,
               const glm::vec3& scale,
               std::vector<glm::vec3> coords,
               const nextfloor::objects::PolygonFactory& factory);
    ~CubeBorder() final = default;

    CubeBorder(CubeBorder&&) = default;
    CubeBorder& operator=(CubeBorder&&) = default;
    CubeBorder(const CubeBorder&) = delete;
    CubeBorder& operator=(const CubeBorder&) = delete;

    std::vector<glm::vec3> getCoordsModelMatrixComputed() const final;
    void ComputeNewLocation() final;
    bool IsObstacleInCollisionAfterPartedMove(const Border& obstacle, float move_part) const final;

    glm::vec3 location() const final { return cube_->location(); }
    /* Coords are a 2.0f width cube, so dimension is 2 * scale */
    glm::vec3 dimension() const final { return 2.0f * cube_->scale(); }
    glm::vec3 movement() const final { return cube_->movement(); }
    float move_factor() const final { return cube_->move_factor(); }
    bool IsMoved() const noexcept final { return cube_->IsMoved(); }
    float diagonal() const noexcept final { return glm::length(dimension()); }

    void set_movement(const glm::vec3& movement) final { cube_->set_movement(movement); }
    void set_move_factor(float move_factor) final { cube_->set_move_factor(move_factor); }

    glm::vec3 getFirstPoint() const final;
    glm::vec3 getLastPoint() const final;

private:
    float CalculateWidth() const final;
    float CalculateHeight() const final;
    float CalculateDepth() const final;
    glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part) const final;

    glm::vec3 scale() const { return cube_->scale(); }

    glm::mat4 CalculateModelMatrix() const;
    void MoveLocation() { cube_->MoveLocation(); }
    void ComputesModelMatrixCoords();

    bool IsObstacleInSameWidthAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const;
    bool IsObstacleInSameHeightAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const;
    bool IsObstacleInSameDepthAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const;

    std::unique_ptr<nextfloor::objects::Polygon> cube_{nullptr};
    std::vector<glm::vec3> coords_;
    /** Border coords in Model Matrix */
    std::vector<glm::vec3> coords_model_matrix_computed_;
};

}  // namespace physics

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSICS_BORDER_H_
