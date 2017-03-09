/*
* Box class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/box.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>
#include <vector>

namespace engine {
namespace geometry {

/* Constructors */
Box::Box()
    :Box(1.0f, glm::vec4(0.0f), {{}}) {}

Box::Box(float scale, glm::vec4 location, std::vector<glm::vec3> coords)
        : Cube(scale, location)
{
    coords_ = coords;
}

Box::Box(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords)
        : Cube(scale, location, move)
{
    coords_ = coords;
}

/* Compute coordinates for current box */
std::vector<glm::vec3> Box::ComputeCoords() const
{
    std::vector<glm::vec3> ret;
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(location_)) * glm::scale(scale_);

    for (auto &point : coords_) {
        ret.push_back(glm::vec3(model * glm::vec4(point, 1.0f)));
    }

    return ret;
}

/* Compute new location for current box on the next move */
void Box::MoveCoords()
{
    if (!IsMoved()) {
        distance_ = -1.0f;
        return;
    }

    /* if distance_ is 15, we change room.
       Dont change Y, because we dont move beside or below room.
       Ugly stuff: to be improved */
    if (distance_ == 15.0f) {
        location_[0] += move_[0] * distance_;
        location_[2] += move_[2] * distance_;
    }
    else if (distance_ != -1.0f) {
        location_ += distance_ * move_;
        move_ = -move_;
    } else {
        location_ += move_;
    }

    distance_ = -1.0f;
}

}//namespace geometry
}//namespace engine
