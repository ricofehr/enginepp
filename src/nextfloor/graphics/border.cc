/**
 *  @file border.cc
 *  @brief Border class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/graphics/border.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <tbb/tbb.h>
#include <memory>

namespace nextfloor {

namespace graphics {

namespace {

/* Default base coords for the box */
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


Border::Border(glm::vec3 scale, glm::vec4 location)
    :Border(scale, location, glm::vec4(0.0f), sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move)
    :Border(glm::vec3(scale), location, move) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move)
    :Border(scale, location, move, sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Border(glm::vec3(scale), location, move, coords) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
{
    cube_ = std::make_unique<Cube>(scale, location, move);
    coords_ = coords;
}

std::vector<glm::vec3> Border::GetCoordsModelMatrixComputed() const
{
    std::vector<glm::vec3> coords_model_matrix_computed(coords_.size());
    glm::mat4 model_matrix = GetModelMatrix();

    /* Parallell coords compute with tbb */
    tbb::parallel_for (0, static_cast<int>(coords_.size()), 1, [&](int i) {
        unsigned long index = static_cast<unsigned long>(i);
        coords_model_matrix_computed[index] = glm::vec3(model_matrix * glm::vec4(coords_[index], 1.0f));
    });

    return coords_model_matrix_computed;
}

glm::mat4 Border::GetModelMatrix() const
{
    return glm::translate(glm::mat4(1.0f), glm::vec3(location())) * glm::scale(scale());
}

void Border::ComputeNewLocation()
{
    if (!IsMoved()) {
        set_distance(1.0f);
        return;
    }

    /* Compute new location coords for border */
    MoveLocation();
}

} // namespace graphics

} // namespace nextfloor
