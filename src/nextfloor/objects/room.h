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

namespace nextfloor {

namespace objects {

/**
 *  @class Room
 *  @brief Define a Room, inherits Model abstract class
 */
class Room : public ModelMesh {

public:
    Room(const glm::vec3& location);
    ~Room() final = default;

    void InitChildsIntoGrid() final;

private:
    void AddRock(const glm::vec3& relative_location);
    void AddWalls();
    void AddFrontWall();
    void AddRightWall();
    void AddBackWall();
    void AddLeftWall();
    void AddFloor();
    void AddRoof();
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROOM_H_
