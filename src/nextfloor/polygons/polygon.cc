/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/polygon.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "nextfloor/renderer/game_window.h"
#include "nextfloor/objects/camera.h"

namespace nextfloor {

namespace polygons {

glm::vec3 Polygon::move() const {
    using nextfloor::renderer::GameWindow;
    return move_ * GameWindow::getMoveFactor();
}

void Polygon::UpdateModelViewProjectionMatrix()
{
    glm::mat4 projection_matrix = GetProjectionMatrix();
    glm::mat4 view_matrix = GetViewMatrix();
    glm::mat4 model_matrix = GetModelMatrix();

    /* Our ModelViewProjection : multiplication of our 3 matrices */
    mvp_ = projection_matrix * view_matrix * model_matrix * glm::scale(scale_);
}

glm::mat4 Polygon::GetProjectionMatrix()
{
    using nextfloor::renderer::GameWindow;

    auto camera = GameWindow::getCamera();
    glm::mat4 projection_matrix = glm::perspective(glm::radians(camera->fov()),
                                            GameWindow::getWidth() / GameWindow::getHeight(),
                                            0.1f, 300.0f);
    return projection_matrix;
}

glm::mat4 Polygon::GetViewMatrix()
{
    using nextfloor::renderer::GameWindow;

    auto camera = GameWindow::getCamera();
    glm::mat4 view_matrix = glm::lookAt(
        camera->location(),
        camera->location() + camera->direction(),
        camera->head()
    );

    return view_matrix;
}

glm::mat4 Polygon::GetModelMatrix()
{
    return glm::translate(glm::mat4(1.0f), location_);
}

} // namespace graphics

} // namespace nextfloor
