/**
 *  @file room.h
 *  @brief Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOM_H_
#define NEXTFLOOR_OBJECTS_ROOM_H_

#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/objects/model_mesh.h"
#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Room
 *  @brief Define a Room, inherits Model abstract class
 */
class Room : public ModelMesh {

public:
    Room(std::unique_ptr<Grid> grid, std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> childs);
    ~Room() final = default;

    void InitChildsIntoGrid() final;

private:
    void InitChilds(std::vector<std::unique_ptr<Mesh>> childs);
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROOM_H_
