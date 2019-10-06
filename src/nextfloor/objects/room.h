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
    Room(const glm::vec3& location, const MeshFactory& factory);
    ~Room() final = default;

    void InitChildsIntoGrid() final;

private:
    void AddRock(const glm::vec3& relative_location, const MeshFactory& factory);
    void AddWalls(const MeshFactory& factory);
    void AddFrontWall(const MeshFactory& factory);
    void AddRightWall(const MeshFactory& factory);
    void AddBackWall(const MeshFactory& factory);
    void AddLeftWall(const MeshFactory& factory);
    void AddFloor(const MeshFactory& factory);
    void AddRoof(const MeshFactory& factory);
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROOM_H_
