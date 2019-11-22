/**
 *  @file wall.h
 *  @brief Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_WALL_H_
#define NEXTFLOOR_PLAYGROUND_WALL_H_

#include "nextfloor/mesh/drawing_mesh.h"

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace playground {

/**
 *  @class Wall
 *  @brief Wall : define Abstract Wall class
 */
class Wall : public nextfloor::mesh::DrawingMesh {

public:
    ~Wall() override = default;

    virtual void AddDoor() = 0;
    virtual void AddWindow() = 0;

    void AddBricks(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const final;
    std::string class_name() final { return "Wall"; }

protected:
    glm::vec3 brick_dimension_{0.0f};
    glm::ivec3 bricks_count_{0, 0, 0};
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
