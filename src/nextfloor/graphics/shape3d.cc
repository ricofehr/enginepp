/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/graphics/shape3d.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "nextfloor/universe/dynamic/camera.h"
#include "nextfloor/core/config_engine.h"

namespace nextfloor {

namespace graphics {

float Shape3D::sMoveFactor = 1.0f;

void Shape3D::ComputeMVP()
{
    /* width and height config values */
    using nextfloor::core::ConfigEngine;
    float window_width = ConfigEngine::getSetting<float>("width");
    float window_height = ConfigEngine::getSetting<float>("height");

    /* Get active Camera */
    using nextfloor::universe::dynamic::Camera;
    auto cam = Camera::active();

    /* Projection Matrix */
    glm::mat4 projection = glm::perspective(glm::radians(cam->fov()),
                                            window_width / window_height,
                                            0.1f, 300.0f);

    /* View Matrix */
    glm::mat4 view = glm::lookAt(
        cam->location(),
        cam->location() + cam->direction(),
        cam->head()
    );

    /* Compute New location_ coords */
    MoveLocation();

    /* Model Matrix */
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location_));

    /* Our ModelViewProjection : multiplication of our 3 matrices */
    mvp_ = projection * view * model * glm::scale(scale_);
}

} // namespace graphics

} // namespace nextfloor
