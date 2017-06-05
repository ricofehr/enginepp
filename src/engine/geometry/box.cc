/*
* Box class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/box.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <cilk/cilk.h>

namespace engine {
namespace geometry {

namespace {
static const std::vector<glm::vec3> sDefaultCoords = {
    /* Front */
    {-1.0f,  1.0f,  1.0f},
    { 1.0f,  1.0f,  1.0f},
    { 1.0f, -1.0f,  1.0f},
    {-1.0f, -1.0f,  1.0f},
    /* Back */
    {-1.0f,  1.0f, -1.0f},
    { 1.0f,  1.0f, -1.0f},
    { 1.0f, -1.0f, -1.0f},
    {-1.0f, -1.0f, -1.0f},
    /* Left */
    {-1.0f,  1.0f,  1.0f},
    {-1.0f,  1.0f, -1.0f},
    {-1.0f, -1.0f, -1.0f},
    {-1.0f, -1.0f,  1.0f},
    /* Right */
    { 1.0f,  1.0f,  1.0f},
    { 1.0f,  1.0f, -1.0f},
    { 1.0f, -1.0f, -1.0f},
    { 1.0f, -1.0f,  1.0f},
    /* Top */
    {-1.0f,  1.0f,  1.0f},
    {-1.0f,  1.0f, -1.0f},
    { 1.0f,  1.0f, -1.0f},
    { 1.0f,  1.0f,  1.0f},
    /* Bottom */
    {-1.0f, -1.0f,  1.0f},
    {-1.0f, -1.0f, -1.0f},
    { 1.0f, -1.0f, -1.0f},
    { 1.0f, -1.0f,  1.0f},

};
}

/* Constructors */
Box::Box()
    :Box(1.0f, glm::vec4(0.0f), sDefaultCoords) {}

Box::Box(float scale, glm::vec4 location)
    :Box(glm::vec3(scale), location) {}

Box::Box(glm::vec3 scale, glm::vec4 location)
    :Box(scale, location, glm::vec4(0.0f), sDefaultCoords) {}

Box::Box(float scale, glm::vec4 location, glm::vec4 move)
    :Box(glm::vec3(scale), location, move) {}

Box::Box(glm::vec3 scale, glm::vec4 location, glm::vec4 move)
    :Box(scale, location, move, sDefaultCoords) {}

Box::Box(float scale, glm::vec4 location, std::vector<glm::vec3> coords)
        : Box(glm::vec3(scale), location, glm::vec4(0.0f), coords) {}

Box::Box(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords)
        : Box(scale, location, glm::vec4(0.0f), coords) {}

Box::Box(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Box(glm::vec3(scale), location, move, coords) {}

Box::Box(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Cube(scale, location, move)
{
    coords_ = coords;
}

/* Compute coordinates for current box */
std::vector<glm::vec3> Box::ComputeCoords() const
{
    std::vector<glm::vec3> ret(coords_.size());
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location_)) * glm::scale(scale_);

    /* Parallell coords compute with cilkplus */
    cilk_for (auto i = 0; i < coords_.size(); i++) {
        ret[i] = glm::vec3(model * glm::vec4(coords_[i], 1.0f));
    }

    return ret;
}

/* Compute new location for current box on the next move */
void Box::MoveCoords()
{
    if (!IsMoved()) {
        distance_ = -1.0f;
        return;
    }

    if (distance_ != -1.0f) {
        location_ += distance_ * move_ * sMoveFactor;
        move_ = -move_;
    } else {
        location_ += move_ * sMoveFactor;
    }

    distance_ = -1.0f;
}

}//namespace geometry
}//namespace engine
