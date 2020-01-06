/**
 *  @file wall.h
 *  @brief Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_WALL_H_
#define NEXTFLOOR_PLAYGROUND_WALL_H_

#include "nextfloor/mesh/drawing_mesh.h"

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace playground {

/**
 *  @class Wall
 *  @brief Wall : define Abstract Wall class
 */
class Wall : public nextfloor::mesh::CompositeMesh {

public:
    ~Wall() override = default;

    virtual void AddDoor() = 0;
    virtual void AddWindow() = 0;

    void AddBricks(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    std::vector<nextfloor::mesh::Mesh*> FindCollisionNeighborsOf(const nextfloor::mesh::Mesh& target) const final;
    std::unique_ptr<nextfloor::mesh::Mesh> remove_child(nextfloor::mesh::Mesh* child) final;

    std::string class_name() final { return "Wall"; }
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
