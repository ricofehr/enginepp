/*
 *   Border class file
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/graphics/border.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <cilk/cilk.h>

namespace engine {

namespace graphics {

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
} // anonymous namespace

Border::Border()
    :Border(1.0f, glm::vec4(0.0f), sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location)
    :Border(glm::vec3(scale), location) {}

Border::Border(glm::vec3 scale, glm::vec4 location)
    :Border(scale, location, glm::vec4(0.0f), sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move)
    :Border(glm::vec3(scale), location, move) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move)
    :Border(scale, location, move, sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location, std::vector<glm::vec3> coords)
        : Border(glm::vec3(scale), location, glm::vec4(0.0f), coords) {}

Border::Border(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords)
        : Border(scale, location, glm::vec4(0.0f), coords) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Border(glm::vec3(scale), location, move, coords) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
{
    cube_ = std::make_unique<Cube>(scale, location, move);
    coords_ = coords;
}

std::vector<glm::vec3> Border::ComputeCoords() const
{
    std::vector<glm::vec3> ret(coords_.size());
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location())) * glm::scale(scale());

    /* Parallell coords compute with cilkplus */
    cilk_for (auto i = 0; i < coords_.size(); i++) {
        ret[i] = glm::vec3(model * glm::vec4(coords_[i], 1.0f));
    }

    return ret;
}

void Border::MoveCoords()
{
    if (!IsMoved()) {
        set_distance(-1.0f);
        return;
    }

    /* Compute new location coords for border */
    MoveLocation();
}

} // namespace graphics

} // namespace engine
