/**
 *  @file border.cc
 *  @brief Border class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/border.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

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


Border::Border(glm::vec3 location, glm::vec3 scale)
    :Border(location, scale, sDefaultCoords) {}

Border::Border(glm::vec3 location, float scale)
    :Border(location, glm::vec3(scale), sDefaultCoords) {}

Border::Border(glm::vec3 location, float scale, std::vector<glm::vec3> coords)
        : Border(location, glm::vec3(scale), coords) {}

Border::Border(glm::vec3 location, glm::vec3 scale, std::vector<glm::vec3> coords)
{
    using nextfloor::core::CommonServices;
    cube_ = CommonServices::getFactory()->MakeCube(location, glm::vec3(scale));
    coords_ = coords;
    ComputesModelMatrixCoords();
}

std::vector<glm::vec3> Border::getCoordsModelMatrixComputed() const
{
    return coords_model_matrix_computed_;
}

void Border::ComputesModelMatrixCoords()
{
    glm::mat4 model_matrix = CalculateModelMatrix();

    coords_model_matrix_computed_.resize(coords_.size());

    /* Parallell coords compute with tbb */
    tbb::parallel_for (0, static_cast<int>(coords_.size()), 1, [&](int i) {
        unsigned long index = static_cast<unsigned long>(i);
        coords_model_matrix_computed_[index] = glm::vec3(model_matrix * glm::vec4(coords_[index], 1.0f));
    });
}

float Border::CalculateWidth()
{
    return coords_model_matrix_computed_.at(1).x - getFirstPoint().x;
}

float Border::CalculateHeight()
{
    return coords_model_matrix_computed_.at(3).y - getFirstPoint().y;
}

float Border::CalculateDepth()
{
    return coords_model_matrix_computed_.at(4).z - getFirstPoint().z;
}

glm::vec3 Border::getFirstPoint()
{
    return coords_model_matrix_computed_.at(0);
}

bool Border::IsObstacleInCollisionAfterPartedMove(nextfloor::objects::EngineBorder* obstacle, float move_part)
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

bool Border::IsObstacleInSameWidthAfterPartedMove(nextfloor::objects::EngineBorder* obstacle, float move_part)
{
    auto current_x_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).x;
    auto obstacle_x_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).x;

    if (current_x_afer_parted_move <= obstacle_x_afer_parted_move + obstacle->CalculateWidth() &&
        obstacle_x_afer_parted_move <= current_x_afer_parted_move + CalculateWidth()) {
        return true;
    }

    return false;
}

bool Border::IsObstacleInSameHeightAfterPartedMove(nextfloor::objects::EngineBorder* obstacle, float move_part)
{
    auto current_y_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).y;
    auto obstacle_y_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).y;

    if (current_y_afer_parted_move >= obstacle_y_afer_parted_move + obstacle->CalculateHeight() &&
        obstacle_y_afer_parted_move >= current_y_afer_parted_move + CalculateHeight()) {
        return true;
    }

    return false;
}

bool Border::IsObstacleInSameDepthAfterPartedMove(nextfloor::objects::EngineBorder* obstacle, float move_part)
{
    auto current_z_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).z;
    auto obstacle_z_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).z;

    if (current_z_afer_parted_move >= obstacle_z_afer_parted_move + obstacle->CalculateDepth() &&
        obstacle_z_afer_parted_move >= current_z_afer_parted_move + CalculateDepth()) {
        return true;
    }

    return false;
}

glm::vec3 Border::RetrieveFirstPointAfterPartedMove(float move_part)
{
    return getFirstPoint() + move_part * glm::vec3(move().x, move().y, move().z);
}

glm::mat4 Border::CalculateModelMatrix() const
{
    return glm::translate(glm::mat4(1.0f), location()) * glm::scale(scale());
}

void Border::ComputeNewLocation()
{
    if (!IsMoved()) {
        set_distance(1.0f);
        return;
    }

    /* Compute new location coords for border */
    MoveLocation();
    ComputesModelMatrixCoords();
}

} // namespace graphics

} // namespace nextfloor
