/*
* Shape3D class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/shape3d.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "engine/universe/camera.h"
#include "engine/core/config_engine.h"

namespace engine {

namespace geometry {

float Shape3D::sMoveFactor = 1.0f;

/*
*   ComputeMVP  -   Compute the ModelViewProjection matrix
*                   for current object.
*/
void Shape3D::ComputeMVP(engine::universe::Camera *cam)
{
    /* width and height config values */
    using engine::core::ConfigEngine;
    float window_width = ConfigEngine::getSetting<float>("width");
    float window_height = ConfigEngine::getSetting<float>("height");

    glm::mat4 projection = glm::perspective(glm::radians(cam->fov()),
                                            window_width / window_height,
                                            0.1f, 300.0f);

    glm::mat4 view = glm::lookAt(
        cam->location(),
        cam->location() + cam->direction(),
        cam->head()
    );

    if (distance_ != -1.0f) {
        location_ += move_ * distance_ * Shape3D::sMoveFactor;
        move_ = -move_;
    } else {
        location_ += move_ * Shape3D::sMoveFactor;
    }
    distance_ = -1.0f;

    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location_));

    /* Our ModelViewProjection : multiplication of our 3 matrices */
    mvp_ = projection * view * model * glm::scale(scale_);
}

}//namespace geometry
}//namespace engine
