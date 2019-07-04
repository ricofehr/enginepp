/**
 *  @file border.cc
 *  @brief Border class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/border.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <tbb/tbb.h>
#include <memory>

namespace nextfloor {

namespace objects {

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

Border::Border(float scale, glm::vec4 location)
    :Border(glm::vec3(scale), location, glm::vec4(1.0f)) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move)
    :Border(glm::vec3(scale), location, move) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move)
    :Border(scale, location, move, sDefaultCoords) {}

Border::Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Border(glm::vec3(scale), location, move, coords) {}

Border::Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
{
    using nextfloor::polygons::Cube;
    cube_ = std::make_unique<Cube>(scale, location);//, move);
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

glm::vec3 Border::RetrieveFirstPointAfterPartedMove(float move_part)
{
    return getFirstPoint() + move_part * glm::vec3(move().x, move().y, move().z);
}

bool Border::IsObstacleInCollisionAfterPartedMove(EngineBorder* obstacle, float move_part)
{
    if (!IsObstacleInSameWidthAfterPartedMove(dynamic_cast<Border*>(obstacle), move_part)) {
        return false;
    }

    if (!IsObstacleInSameHeightAfterPartedMove(dynamic_cast<Border*>(obstacle), move_part)) {
        return false;
    }

    if (!IsObstacleInSameDepthAfterPartedMove(dynamic_cast<Border*>(obstacle), move_part)) {
        return false;
    }

    return true;
}

bool Border::IsObstacleInSameWidthAfterPartedMove(Border* obstacle, float move_part)
{
    auto current_x_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).x;
    auto obstacle_x_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).x;

    if (current_x_afer_parted_move <= obstacle_x_afer_parted_move + obstacle->CalculateWidth() &&
        obstacle_x_afer_parted_move <= current_x_afer_parted_move + CalculateWidth()) {
        return true;
    }

    return false;
}

bool Border::IsObstacleInSameHeightAfterPartedMove(Border* obstacle, float move_part)
{
    auto current_y_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).y;
    auto obstacle_y_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).y;

    if (current_y_afer_parted_move >= obstacle_y_afer_parted_move + obstacle->CalculateHeight() &&
        obstacle_y_afer_parted_move >= current_y_afer_parted_move + CalculateHeight()) {
        return true;
    }

    return false;
}

bool Border::IsObstacleInSameDepthAfterPartedMove(Border* obstacle, float move_part)
{
    auto current_z_afer_parted_move = RetrieveFirstPointAfterPartedMove(move_part).z;
    auto obstacle_z_afer_parted_move = obstacle->RetrieveFirstPointAfterPartedMove(move_part).z;

    if (current_z_afer_parted_move >= obstacle_z_afer_parted_move + obstacle->CalculateDepth() &&
        obstacle_z_afer_parted_move >= current_z_afer_parted_move + CalculateDepth()) {
        return true;
    }

    return false;
}

glm::mat4 Border::CalculateModelMatrix() const
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
    ComputesModelMatrixCoords();
}

} // namespace graphics

} // namespace nextfloor
