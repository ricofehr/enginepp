/*
* Room class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>
#include <memory>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <tbb/mutex.h>

#include "engine/universe/model3d.h"
#include "engine/physics/collision_engine.h"
#include "engine/universe/wall.h"
#include "engine/universe/brick.h"

namespace engine {
namespace universe {

/* Define a Room (with walls and moving objects) in the 3d universe */
class Room : public Model3D {

public:

    /* Constructors */
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows);

    /* Mutators */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

    /* New hop functions */
    void Draw(Camera *cam);
    /* Object Generation functions */
    void GenerateRandomObject();
    void GenerateWalls();
    void GenerateObjects();

private:

    /* Grid Constants For Romm Setting */
    static constexpr int kGRID_Y = 6;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_Y = 2.0f;
    static constexpr float kGRID_UNIT_X = 2.0f;
    static constexpr float kGRID_UNIT_Z = 2.0f;

    /* Some internal functions */
    Model3D *GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale);

    /* Room attributes */
    tbb::mutex room_mutex_;
    std::vector<bool> doors_{false, false, false, false, false, false};
    std::vector<bool> windows_{false, false, false, false, false, false};
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
