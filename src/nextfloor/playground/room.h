/**
 *  @file room.h
 *  @brief Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_ROOM_H_
#define NEXTFLOOR_PLAYGROUND_ROOM_H_

#include "nextfloor/playground/ground.h"

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/playground/grid.h"
#include "nextfloor/playground/wall.h"
#include "nextfloor/mesh/border.h"
#include "nextfloor/mesh/dynamic_mesh.h"

namespace nextfloor {

namespace playground {

/**
 *  @class Room
 *  @brief Define a Room, inherits Model abstract class
 */
class Room : public Ground {

public:
    Room(std::unique_ptr<Grid> grid,
         std::unique_ptr<nextfloor::mesh::Border> border,
         std::vector<std::unique_ptr<Wall>> walls,
         std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects);
    ~Room() final = default;

private:
    void InitChilds(std::vector<std::unique_ptr<Wall>> walls,
                    std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects);
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_ROOM_H_
