/**
 *  @file cube_border.h
 *  @brief CubeBorder class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_CUBEBORDER_H_
#define NEXTFLOOR_PHYSIC_CUBEBORDER_H_

#include "nextfloor/mesh/border.h"

#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace nextfloor {

namespace physic {

/**
 *  @class CubeBorder
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class CubeBorder : public nextfloor::mesh::Border {

public:
    CubeBorder(const glm::vec3& location, const glm::vec3& scale);
    ~CubeBorder() final = default;

    CubeBorder(CubeBorder&&) = default;
    CubeBorder& operator=(CubeBorder&&) = default;
    CubeBorder(const CubeBorder&) = delete;
    CubeBorder& operator=(const CubeBorder&) = delete;

    std::vector<glm::vec3> getCoordsModelMatrixComputed() const final;
    void ComputeNewLocation() final;
    bool IsObstacleInCollisionAfterPartedMove(const Border& obstacle, float move_part) const final;

    /* Coords are a 2.0f width cube, so dimension is 2 * scale */
    glm::vec3 dimension() const final { return 2.0f * scale(); }
    bool IsMoved() const final { return movement_[0] != 0.0f || movement_[1] != 0.0f || movement_[2] != 0.0f; }
    float diagonal() const final { return glm::length(dimension()); }

    glm::vec3 movement() const final { return movement_; }
    float move_factor() const final { return fabs(move_factor_); }
    glm::vec3 location() const final { return location_; }

    void set_move_factor(float move_factor) final { move_factor_ = move_factor; }
    void set_movement(const glm::vec3& movement) final { movement_ = movement; }

    glm::vec3 getFirstPoint() const final;
    glm::vec3 getLastPoint() const final;

private:
    static constexpr float kPoinstStep = 0.10f;
    static constexpr float kInitMoveFactor = 1.0f;

    float CalculateWidth() const final;
    float CalculateHeight() const final;
    float CalculateDepth() const final;
    glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part) const final;

    glm::vec3 scale() const { return scale_; }
    glm::mat4 CalculateModelMatrix() const;

    void MoveLocation()
    {
        location_ += movement() * move_factor();

        if (move_factor_ <= 0.0f) {
            InverseMove();
        }

        move_factor_ = 1.0f;
    }

    void InverseMove() { movement_ = -movement_; }

    void ComputesModelMatrixCoords();

    bool IsObstacleInSameWidthAfterPartedMove(const nextfloor::mesh::Border& obstacle, float move_part) const;
    bool IsObstacleInSameHeightAfterPartedMove(const nextfloor::mesh::Border& obstacle, float move_part) const;
    bool IsObstacleInSameDepthAfterPartedMove(const nextfloor::mesh::Border& obstacle, float move_part) const;

    glm::vec3 location_{0.0f, 0.0f, 0.0f};
    glm::vec3 scale_{0.0f, 0.0f, 0.0f};
    glm::vec3 movement_{0.0f, 0.0f, 0.0f};

    /** MOve factor with collision shape (1 -> no collision detected) */
    float move_factor_{1.0f};

    // std::unique_ptr<nextfloor::mesh::HiddenObject> hidden_cube_{nullptr};
    std::vector<glm::vec3> coords_;
    /** Border coords in Model Matrix */
    std::vector<glm::vec3> coords_model_matrix_computed_;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_BORDER_H_
