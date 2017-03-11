/*
* Shape3D class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/shape3d.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "engine/helpers/proxygl.h"
#include "engine/universe/camera.h"

namespace engine {

namespace geometry {

/*
*   ComputeMVP  -   Compute the ModelViewProjection matrix
*                   for current object.
*
*   Return a mvp mat4 matrix
*/
void Shape3D::ComputeMVP(engine::universe::Camera *cam)
{
    /* kWidthWindow, kHeightWindow  fixed values */
    using engine::helpers::proxygl::kWidthWindow;
    using engine::helpers::proxygl::kHeightWindow;

    glm::mat4 projection = glm::perspective(glm::radians(cam->fov()),
                                            kWidthWindow / kHeightWindow,
                                            0.1f, 300.0f);

    glm::mat4 view = glm::lookAt(
        cam->location(),
        cam->location() + cam->direction(),
        cam->head()
    );

    /* if distance_ is 15, we change room.
       Dont change Y, because we dont move beside or below room.
       Ugly stuff: to be improved */
    if (distance_ == 15.0f) {
        location_[0] += move_[0] * distance_;
        location_[2] += move_[2] * distance_;
    }
    else if (distance_ != -1.0f) {
        location_ += move_ * distance_;
        move_ = -move_;
    } else {
        location_ += move_;
    }
    distance_ = -1.0f;

    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location_));

    /* Our ModelViewProjection : multiplication of our 3 matrices */
    mvp_ = projection * view * model * glm::scale(scale_);
}

}//namespace geometry
}//namespace engine
