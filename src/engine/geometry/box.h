/*
* Box class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GEOMETRY_BOX_H_
#define ENGINE_GEOMETRY_BOX_H_

#include <glm/glm.hpp>

#include <vector>

#include "engine/geometry/cube.h"

namespace engine {
namespace geometry {

/**
*    Each 3d model in the scene must be fill into a box.
*    Uses for coordinates and collision compute.
*    A box is a cube with additional properties, so it ihnerits from Cube
**/
class Box : public Cube {

public:

    Box();
    Box(float scale, glm::vec4 location);
    Box(glm::vec3 scale, glm::vec4 location);
    Box(float scale, glm::vec4 location, glm::vec4 move);
    Box(glm::vec3 scale, glm::vec4 location, glm::vec4 move);
    Box(float scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Box(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords);
    Box(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);
    Box(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    /* Default move and copy constructor / operator */
    Box(Box&&) = default;
    Box& operator=(Box&&) = default;

    Box(const Box&) = default;
    Box& operator=(const Box&) = default;

    /* Default destructor */
    ~Box() override = default;

    std::vector<glm::vec3> ComputeCoords() const;
    void MoveCoords();

private:

    std::vector<glm::vec3> coords_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_GEOMETRY_BOX_H_

