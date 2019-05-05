/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/graphics/shape3d.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "nextfloor/renderer/game_window.h"
#include "nextfloor/universe/dynamic/camera.h"

namespace nextfloor {

namespace graphics {

glm::vec4 Shape3D::move() const {
    using nextfloor::renderer::GameWindow;
    return move_ * GameWindow::getMoveFactor();
}

void Shape3D::ComputeMVP()
{
    using nextfloor::renderer::GameWindow;
    using nextfloor::universe::dynamic::Camera;

    nextfloor::universe::dynamic::Camera* camera = GameWindow::getCamera();

    /* Projection Matrix */
    glm::mat4 projection = glm::perspective(glm::radians(camera->fov()),
                                            GameWindow::getWidth() / GameWindow::getHeight(),
                                            0.1f, 300.0f);

    /* View Matrix */
    glm::mat4 view = glm::lookAt(
        camera->location(),
        camera->location() + camera->direction(),
        camera->head()
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
