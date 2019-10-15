/**
 *  @file border.cc
 *  @brief Border class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/cube_border.h"

/* Need Experimental flag for transform methods */
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

namespace {

/* Default base coords for the box */
static const std::vector<glm::vec3> sDefaultCoords = {
  /* Front */
  {-1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, 1.0f},
  {1.0f, -1.0f, 1.0f},
  {-1.0f, -1.0f, 1.0f},
  /* Back */
  {-1.0f, 1.0f, -1.0f},
  {1.0f, 1.0f, -1.0f},
  {1.0f, -1.0f, -1.0f},
  {-1.0f, -1.0f, -1.0f},
  /* Left */
  {-1.0f, 1.0f, 1.0f},
  {-1.0f, 1.0f, -1.0f},
  {-1.0f, -1.0f, -1.0f},
  {-1.0f, -1.0f, 1.0f},
  /* Right */
  {1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, -1.0f},
  {1.0f, -1.0f, -1.0f},
  {1.0f, -1.0f, 1.0f},
  /* Top */
  {-1.0f, 1.0f, 1.0f},
  {-1.0f, 1.0f, -1.0f},
  {1.0f, 1.0f, -1.0f},
  {1.0f, 1.0f, 1.0f},
  /* Bottom */
  {-1.0f, -1.0f, 1.0f},
  {-1.0f, -1.0f, -1.0f},
  {1.0f, -1.0f, -1.0f},
  {1.0f, -1.0f, 1.0f},

};

}  // anonymous namespace


CubeBorder::CubeBorder(const glm::vec3& location, const glm::vec3& scale) : CubeBorder(location, scale, sDefaultCoords)
{}

CubeBorder::CubeBorder(const glm::vec3& location, const glm::vec3& scale, std::vector<glm::vec3> coords)
{
    location_ = location;
    scale_ = scale;
    coords_ = coords;
    ComputesModelMatrixCoords();
}

std::vector<glm::vec3> CubeBorder::getCoordsModelMatrixComputed() const
{
    return coords_model_matrix_computed_;
}

void CubeBorder::ComputesModelMatrixCoords()
{
    glm::mat4 model_matrix = CalculateModelMatrix();

    coords_model_matrix_computed_.resize(coords_.size());

    /* Parallell coords compute with tbb */
    tbb::parallel_for(0, static_cast<int>(coords_.size()), 1, [&](int i) {
        unsigned long index = static_cast<unsigned long>(i);
        coords_model_matrix_computed_[index] = glm::vec3(model_matrix * glm::vec4(coords_[index], 1.0f));
    });
}

float CubeBorder::CalculateWidth() const
{
    return coords_model_matrix_computed_.at(1).x - getFirstPoint().x;
}

float CubeBorder::CalculateHeight() const
{
    return coords_model_matrix_computed_.at(3).y - getFirstPoint().y;
}

float CubeBorder::CalculateDepth() const
{
    return coords_model_matrix_computed_.at(4).z - getFirstPoint().z;
}

glm::vec3 CubeBorder::getFirstPoint() const
{
    auto first_point = coords_model_matrix_computed_.at(0);
    auto last_point = coords_model_matrix_computed_.at(6);

    /* Add padding */
    if (first_point.x < last_point.x) {
        first_point.x += 0.10f;
    }
    else {
        first_point.x -= 0.10f;
    }

    if (first_point.y < last_point.y) {
        first_point.y += 0.10f;
    }
    else {
        first_point.y -= 0.10f;
    }

    if (first_point.z < last_point.z) {
        first_point.z += 0.10f;
    }
    else {
        first_point.z -= 0.10f;
    }

    return first_point;
}

glm::vec3 CubeBorder::getLastPoint() const
{
    auto first_point = coords_model_matrix_computed_.at(0);
    auto last_point = coords_model_matrix_computed_.at(6);

    /* Add padding */
    if (first_point.x < last_point.x) {
        last_point.x -= 0.10f;
    }
    else {
        last_point.x += 0.10f;
    }

    if (first_point.y < last_point.y) {
        last_point.y -= 0.10f;
    }
    else {
        last_point.y += 0.10f;
    }

    if (first_point.z < last_point.z) {
        last_point.z -= 0.10f;
    }
    else {
        last_point.z += 0.10f;
    }

    return last_point;
}

bool CubeBorder::IsObstacleInCollisionAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const
{
    if (!IsObstacleInSameWidthAfterPartedMove(obstacle, move_part)) {
        return false;
    }

    if (!IsObstacleInSameHeightAfterPartedMove(obstacle, move_part)) {
        return false;
    }

    if (!IsObstacleInSameDepthAfterPartedMove(obstacle, move_part)) {
        return false;
    }

    return true;
}

bool CubeBorder::IsObstacleInSameWidthAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const
{
    auto current_x_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).x;
    auto obstacle_x_afer_parted_move = obstacle.RetrieveFirstPointAfterPartedMove(move_part).x;

    if (current_x_afer_parted_move <= obstacle_x_afer_parted_move + obstacle.CalculateWidth()
        && obstacle_x_afer_parted_move <= current_x_afer_parted_move + CalculateWidth()) {
        return true;
    }

    return false;
}

bool CubeBorder::IsObstacleInSameHeightAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const
{
    auto current_y_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).y;
    auto obstacle_y_afer_parted_move = obstacle.RetrieveFirstPointAfterPartedMove(move_part).y;

    if (current_y_afer_parted_move >= obstacle_y_afer_parted_move + obstacle.CalculateHeight()
        && obstacle_y_afer_parted_move >= current_y_afer_parted_move + CalculateHeight()) {
        return true;
    }

    return false;
}

bool CubeBorder::IsObstacleInSameDepthAfterPartedMove(const nextfloor::objects::Border& obstacle, float move_part) const
{
    auto current_z_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).z;
    auto obstacle_z_afer_parted_move = obstacle.RetrieveFirstPointAfterPartedMove(move_part).z;

    if (current_z_afer_parted_move >= obstacle_z_afer_parted_move + obstacle.CalculateDepth()
        && obstacle_z_afer_parted_move >= current_z_afer_parted_move + CalculateDepth()) {
        return true;
    }

    return false;
}

glm::vec3 CubeBorder::RetrieveFirstPointAfterPartedMove(float move_part) const
{
    return getFirstPoint() + move_part * glm::vec3(movement().x, movement().y, movement().z);
}

glm::mat4 CubeBorder::CalculateModelMatrix() const
{
    return glm::translate(glm::mat4(1.0f), location()) * glm::scale(scale());
}

void CubeBorder::ComputeNewLocation()
{
    if (!IsMoved()) {
        set_move_factor(1.0f);
        return;
    }

    /* Compute new location coords for border */
    MoveLocation();
    ComputesModelMatrixCoords();
}

}  // namespace physics

}  // namespace nextfloor
