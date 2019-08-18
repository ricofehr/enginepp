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

    CubeBorder(glm::vec3 location, glm::vec3 scale);
    CubeBorder(glm::vec3 location, float scale);
    CubeBorder(glm::vec3 location, float scale, std::vector<glm::vec3> coords);
    CubeBorder(glm::vec3 location, glm::vec3 scale, std::vector<glm::vec3> coords);

    CubeBorder(CubeBorder&&) = default;
    CubeBorder& operator=(CubeBorder&&) = default;
    CubeBorder(const CubeBorder&) = delete;
    CubeBorder& operator=(const CubeBorder&) = delete;

    virtual ~CubeBorder() override = default;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const override final;
    virtual void ComputeNewLocation() override final;
    virtual bool IsObstacleInCollisionAfterPartedMove(Border* obstacle, float move_part) override final;

    virtual glm::vec3 location() const override final { return cube_->location(); }
    /* Coords are a 2.0f width cube, so dimension is 2 * scale */
    virtual glm::vec3 dimension() const override final { return 2.0f * cube_->scale(); }
    virtual glm::vec3 movement() const override final { return cube_->movement(); }
    virtual float move_factor() const override final { return cube_->move_factor(); }
    virtual bool IsMoved() const noexcept override final { return cube_->IsMoved(); }
    virtual float diagonal() const noexcept override final { return glm::length(dimension()); }

    virtual void set_movement(glm::vec3 movement) const override final { return cube_->set_movement(movement); }
    virtual void set_move_factor(float move_factor) const override final { return cube_->set_move_factor(move_factor); }

    virtual glm::vec3 getFirstPoint() override;
    virtual glm::vec3 getLastPoint() override;

private:

    virtual float CalculateWidth() override;
    virtual float CalculateHeight() override;
    virtual float CalculateDepth() override;
    virtual glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part) override;

    glm::vec3 scale() const { return cube_->scale(); }

    glm::mat4 CalculateModelMatrix() const;
    void MoveLocation() { cube_->MoveLocation(); }
    void ComputesModelMatrixCoords();

    bool IsObstacleInSameWidthAfterPartedMove(nextfloor::objects::Border* obstacle, float move_part);
    bool IsObstacleInSameHeightAfterPartedMove(nextfloor::objects::Border* obstacle, float move_part);
    bool IsObstacleInSameDepthAfterPartedMove(nextfloor::objects::Border* obstacle, float move_part);

    std::unique_ptr<nextfloor::objects::Polygon> cube_{nullptr};
    std::vector<glm::vec3> coords_;
    /** Border coords in Model Matrix */
    std::vector<glm::vec3> coords_model_matrix_computed_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_BORDER_H_

