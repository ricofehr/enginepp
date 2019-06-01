/**
 *  @file border.h
 *  @brief Border class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRAPHICS_BORDER_H_
#define NEXTFLOOR_GRAPHICS_BORDER_H_

#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "nextfloor/graphics/cube.h"


namespace nextfloor {

namespace graphics {

/**
 *  @class Border
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class Border {

public:

    Border(glm::vec3 scale, glm::vec4 location);

    Border(float scale, glm::vec4 location, glm::vec4 move);

    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move);

    Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    Border(Border&&) = default;

    Border& operator=(Border&&) = default;

    /* Copy constructor Deleted because cube_ is unique_ptr */
    Border(const Border&) = delete;

    /* Copy assignment Deleted because cube_ is unique_ptr */
    Border& operator=(const Border&) = delete;

    ~Border() = default;

    std::vector<glm::vec3> GetCoordsModelMatrixComputed() const;

    void ComputeNewLocation();

    /* Delegate Accessors */
    bool IsMoved() const { return cube_->IsMoved(); }
    int IsMovedX() const { return cube_->IsMovedX(); }
    int IsMovedY() const { return cube_->IsMovedY(); }
    int IsMovedZ() const { return cube_->IsMovedZ(); }
    glm::vec4 location() const { return cube_->location(); }
    glm::vec3 scale() const { return cube_->scale(); }
    glm::vec4 move() const { return cube_->move(); }
    float distance() const { return cube_->distance(); }

    /* Delegate Mutators */
    void set_distance(float distance) { cube_->set_distance(distance); }
    void set_move(glm::vec3 move) { cube_->set_move(move); }
    void InverseMove() { cube_->InverseMove(); }

private:

    glm::mat4 GetModelMatrix() const;

    /* Delegate action */
    void MoveLocation() { cube_->MoveLocation(); }

    /** Design the border */
    std::unique_ptr<Cube> cube_{nullptr};

    /** Border coords */
    std::vector<glm::vec3> coords_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_BORDER_H_

