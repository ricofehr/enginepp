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
#include "engine/universe/camera.h"

namespace engine {
namespace universe {

/* Define a Room (with walls and moving objects) in the 3d universe */
class Room : public Model3D {

public:
    /* Room Side Constants */
    static constexpr int kFLOOR = 0;
    static constexpr int kROOF = 1;
    static constexpr int kLEFT = 2;
    static constexpr int kRIGHT = 3;
    static constexpr int kFRONT = 4;
    static constexpr int kBACK = 5;

    /* Constructors */
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::unique_ptr<Camera> cam);
    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows,
         std::unique_ptr<Camera> cam);

    /* Mutators */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

    /* New hop functions */
    void Draw(Camera *cam);
    void DetectCollision(std::vector<Room*> neighbors);
    /* Object Generation functions */
    void GenerateRandomObject();
    void GenerateWalls();
    void GenerateObjects();
    
    void MoveCamera() { if (cam_ != nullptr) cam_->Move(); };

private:
    /* Some internal functions */
    Model3D *GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale);
    void PivotCollision(Model3D *object, std::vector<Room*> neighbors);

    /* Room attributes */
    tbb::mutex room_mutex_;
    std::vector<bool> doors_{false, false, false, false, false, false};
    std::vector<bool> windows_{false, false, false, false, false, false};
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
