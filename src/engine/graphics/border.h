/*
* Border class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GRAPHICS_BORDER_H_
#define ENGINE_GRAPHICS_BORDER_H_

#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "engine/graphics/cube.h"

namespace engine {
namespace graphics {

/**
*    Each 3d model in the scene must be fill into a border like a cubed box.
*    Uses for coordinates and collision compute.
**/
class Border {

public:

    /* Constructors */
    Border();
    Border(float scale, glm::vec4 location);
    Border(glm::vec3 scale, glm::vec4 location);
    Border(float scale, glm::vec4 location, glm::vec4 move);
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move);
    Border(float scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Border(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    /* Default move constructor / assignment */
    Border(Border&&) = default;
    Border& operator=(Border&&) = default;

    /* Delete copy constructor / assignment because cube_ is unique_ptr */
    Border(const Border&) = delete;
    Border& operator=(const Border&) = delete;

    /* Default destructor */
    ~Border() = default;

    /* Coords Computes */
    std::vector<glm::vec3> ComputeCoords() const;
    void MoveCoords();

    /* Delegate Move */
    void MoveLocation() { cube_->MoveLocation(); }

    /* Delegate Accessors */
    bool IsMoved() const { return cube_->IsMoved(); }
    bool IsMovedX() const { return cube_->IsMovedX(); }
    bool IsMovedY() const { return cube_->IsMovedY(); }
    bool IsMovedZ() const { return cube_->IsMovedZ(); }
    glm::vec4 location() const { return cube_->location(); }
    glm::vec3 scale() const { return cube_->scale(); }
    glm::vec4 move() const { return cube_->move(); }

    /* Delegate Mutators */
    void set_distance(float distance) { cube_->set_distance(distance); }
    void set_move(glm::vec3 move) { cube_->set_move(move); }
    void InverseMove() { cube_->InverseMove(); }

private:

    std::unique_ptr<Cube> cube_{nullptr};
    std::vector<glm::vec3> coords_;
};

}//namespace graphics
}//namespace engine

#endif //ENGINE_GRAPHICS_BORDER_H_

