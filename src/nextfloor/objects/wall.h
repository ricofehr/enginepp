/**
 *  @file wall.h
 *  @brief Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WALL_H_
#define NEXTFLOOR_OBJECTS_WALL_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>
#include <string>


namespace nextfloor {

namespace objects {

/**
 *  @class Wall
 *  @brief Wall : define Abstract Wall class
 */
class Wall : public ModelMesh {

public:
    ~Wall() override = default;

    virtual void AddDoor() = 0;
    virtual void AddWindow() = 0;

protected:
    virtual void AddBricks(glm::vec3 firstpoint, glm::vec3 lastpoint);

    virtual std::string texture_file() const = 0;

    glm::vec3 brick_dimension_{0.0f};
    glm::ivec3 bricks_count_{0, 0, 0};
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
