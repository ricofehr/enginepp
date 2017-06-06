/*
* Border class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GRAPHICS_BORDER_H_
#define ENGINE_GRAPHICS_BORDER_H_

#include <glm/glm.hpp>

#include <vector>

#include "engine/graphics/cube.h"

namespace engine {
namespace graphics {

/**
*    Each 3d model in the scene must be fill into a box.
*    Uses for coordinates and collision compute.
*    A box is a cube with additional properties
**/
class Border : public Cube {

public:

    Border();
    Border(float scale, glm::vec4 location);
    Border(glm::vec3 scale, glm::vec4 location);
    Border(float scale, glm::vec4 location, glm::vec4 move);
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move);
    Border(float scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Border(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    /* Default move and copy constructor / operator */
    Border(Border&&) = default;
    Border& operator=(Border&&) = default;

    Border(const Border&) = default;
    Border& operator=(const Border&) = default;

    /* Default destructor */
    ~Border() override = default;

    std::vector<glm::vec3> ComputeCoords() const;
    void MoveCoords();

private:

    std::vector<glm::vec3> coords_;
};

}//namespace graphics
}//namespace engine

#endif //ENGINE_GRAPHICS_BORDER_H_

