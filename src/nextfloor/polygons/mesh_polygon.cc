/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/mesh_polygon.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <tbb/mutex.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

bool Polygon::is_viewprojection_setted_ = false;

}

namespace polygons {

glm::vec3 MeshPolygon::movement() const
{
    using nextfloor::core::CommonServices;
    return movement_ * CommonServices::getWindowSettings()->getFpsFixMoveFactor();
}

void MeshPolygon::UpdateModelViewProjectionMatrix(const nextfloor::objects::Camera& camera)
{
    static glm::mat4 view_projection_matrix;
    static tbb::mutex mutex;

    mutex.lock();
    if (!is_viewprojection_setted_) {
        view_projection_matrix = GetProjectionMatrix(camera) * GetViewMatrix(camera);
        is_viewprojection_setted_ = true;
    }
    mutex.unlock();

    mvp_ = view_projection_matrix * GetModelMatrix() * glm::scale(scale_);
}

glm::mat4 MeshPolygon::GetProjectionMatrix(const nextfloor::objects::Camera& camera)
{
    using nextfloor::core::CommonServices;

    glm::mat4 projection_matrix = glm::perspective(glm::radians(camera.fov()),
                                                   CommonServices::getWindowSettings()->getWidth()
                                                     / CommonServices::getWindowSettings()->getHeight(),
                                                   0.1f,
                                                   300.0f);
    return projection_matrix;
}

glm::mat4 MeshPolygon::GetViewMatrix(const nextfloor::objects::Camera& camera)
{
    glm::mat4 view_matrix = glm::lookAt(camera.location(), camera.location() + camera.direction(), camera.head());

    return view_matrix;
}

glm::mat4 MeshPolygon::GetModelMatrix()
{
    return glm::translate(glm::mat4(1.0f), location_);
}

}  // namespace polygons

}  // namespace nextfloor
