/*
* Room class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>
#include <tbb/mutex.h>

#include "engine/universe/model3d.h"
#include "engine/parallell/engine_parallell.h"
#include "engine/universe/wall.h"
#include "engine/universe/window_model.h"
#include "engine/universe/door.h"
#include "engine/universe/brick.h"
#include "engine/universe/camera.h"

namespace engine {
namespace universe {

/* Define a Room (with walls and moving objects) in the 3d universe */
class Room {

public:
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::vector<bool> is_doors, Camera *cam,
         engine::parallell::EngineParallell *proxy_parallell);

    glm::vec4 location() const { return location_; }
    int countObjects() const { return objects_.size(); }
    bool IsFull() const { return objects_.size() >= nbobjects_; }

    void Draw();
    void DetectCollision();
    void GenerateRandomObject();
    void ReinitGrid();

private:
    glm::vec4 location_;
    int nbobjects_{33};
    Camera *cam_{nullptr};
    bool grid_[5][15][15]{false};
    std::vector<std::unique_ptr<Wall>> walls_;
    std::vector<std::unique_ptr<Door>> doors_;
    std::vector<std::unique_ptr<WindowModel>> windows_;
    std::vector<std::unique_ptr<Model3D>> objects_;
    engine::parallell::EngineParallell *proxy_parallell_{nullptr};
    tbb::mutex room_mutex_;

    void GenerateObjects();
    void GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale);
    void PivotCollision(Model3D *object);
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
