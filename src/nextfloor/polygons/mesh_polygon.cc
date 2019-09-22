/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/mesh_polygon.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <tbb/mutex.h>

#include "nextfloor/objects/camera.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

bool Polygon::is_viewprojection_setted_ = false;

}

namespace polygons {

glm::vec3 MeshPolygon::movement() const
{
    using nextfloor::core::CommonServices;
    return movement_ * CommonServices::getWindowSettings().getFpsFixMoveFactor();
}

void MeshPolygon::UpdateModelViewProjectionMatrix()
{
    static glm::mat4 view_projection_matrix;
    static tbb::mutex mutex;

    mutex.lock();
    if (!is_viewprojection_setted_) {
        view_projection_matrix = GetProjectionMatrix() * GetViewMatrix();
        is_viewprojection_setted_ = true;
    }
    mutex.unlock();

    mvp_ = view_projection_matrix * GetModelMatrix() * glm::scale(scale_);
}

glm::mat4 MeshPolygon::GetProjectionMatrix()
{
    using nextfloor::core::CommonServices;

    glm::mat4 projection_matrix
      = glm::perspective(glm::radians(CommonServices::getActiveCamera().fov()),
                         CommonServices::getWindowSettings().getWidth() / CommonServices::getWindowSettings().getHeight(),
                         0.1f,
                         300.0f);
    return projection_matrix;
}

glm::mat4 MeshPolygon::GetViewMatrix()
{
    using nextfloor::core::CommonServices;

    glm::mat4 view_matrix
      = glm::lookAt(CommonServices::getActiveCamera().location(),
                    CommonServices::getActiveCamera().location() + CommonServices::getActiveCamera().direction(),
                    CommonServices::getActiveCamera().head());

    return view_matrix;
}

glm::mat4 MeshPolygon::GetModelMatrix()
{
    return glm::translate(glm::mat4(1.0f), location_);
}

}  // namespace polygons

}  // namespace nextfloor
