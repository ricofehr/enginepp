/**
 *  @file border.h
 *  @brief Border class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_BORDER_H_
#define NEXTFLOOR_PHYSICS_BORDER_H_

#include "nextfloor/objects/engine_border.h"

//#define GLM_SWIZZLE
#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "nextfloor/polygons/cube.h"


namespace nextfloor {

namespace physics {

using nextfloor::objects::EngineBorder;

/**
 *  @class Border
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class Border : public EngineBorder {

public:

    Border(glm::vec3 location, glm::vec3 scale);

    Border(glm::vec3 location, float scale);

    Border(glm::vec3 location, float scale, std::vector<glm::vec3> coords);

    Border(glm::vec3 location, glm::vec3 scale, std::vector<glm::vec3> coords);

    Border(Border&&) = default;

    Border& operator=(Border&&) = default;

    /* Copy constructor Deleted because cube_ is unique_ptr */
    Border(const Border&) = delete;

    /* Copy assignment Deleted because cube_ is unique_ptr */
    Border& operator=(const Border&) = delete;

    virtual ~Border() override = default;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const override final;

    virtual void ComputeNewLocation() override final;

    virtual bool IsObstacleInCollisionAfterPartedMove(EngineBorder* obstacle, float move_part) override final;

    /* Delegate Accessors */
    bool IsMoved() const { return cube_->IsMoved(); }
    // int IsMovedX() const { return cube_->IsMovedX(); }
    // int IsMovedY() const { return cube_->IsMovedY(); }
    // int IsMovedZ() const { return cube_->IsMovedZ(); }
    virtual glm::vec3 location() const override final { return cube_->location(); }
    glm::vec3 scale() const { return cube_->scale(); }
    virtual glm::vec3 move() const override final { return cube_->move(); }
    virtual float distance() const override final { return cube_->distance(); }

    /* Delegate Mutators */
    virtual void set_distance(float distance) override final { cube_->set_distance(distance); }
    //virtual void set_move(glm::vec3 move) override final { cube_->set_move(move); }
    virtual void InverseMove() override final { cube_->InverseMove(); }

private:

    glm::mat4 CalculateModelMatrix() const;

    /* Delegate action */
    void MoveLocation() { cube_->MoveLocation(); }

    void ComputesModelMatrixCoords();

    float CalculateWidth();
    float CalculateHeight();
    float CalculateDepth();
    glm::vec3 getFirstPoint();
    glm::vec3 RetrieveFirstPointAfterPartedMove(float move_part);
    bool IsObstacleInSameWidthAfterPartedMove(Border* obstacle, float move_part);
    bool IsObstacleInSameHeightAfterPartedMove(Border* obstacle, float move_part);
    bool IsObstacleInSameDepthAfterPartedMove(Border* obstacle, float move_part);

    /** Design the border */
    std::unique_ptr<nextfloor::objects::EnginePolygon> cube_{nullptr};

    /** Border coords */
    std::vector<glm::vec3> coords_;

    /** Border coords in Model Matrix */
    std::vector<glm::vec3> coords_model_matrix_computed_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_BORDER_H_

