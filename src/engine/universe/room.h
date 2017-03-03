/*
* Room class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>

#include "engine/universe/model3d.h"
#include "engine/helpers/proxycl.h"
#include "engine/universe/wall.h"
#include "engine/universe/window_model.h"
#include "engine/universe/door.h"
#include "engine/universe/brick.h"

namespace engine {
namespace universe {

/* Define a Room (with walls and moving objects) in the 3d universe */
class Room {

public:
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::vector<bool> is_doors, engine::helpers::ProxyCL *proxy_cl);

    glm::vec4 location() const { return location_; }

    void Draw();
    void DetectCollision();

private:
    glm::vec4 location_;
    std::vector<std::unique_ptr<Wall>> walls_;
    std::vector<std::unique_ptr<Door>> doors_;
    std::vector<std::unique_ptr<WindowModel>> windows_;
    std::vector<std::unique_ptr<Model3D>> objects_;
    engine::helpers::ProxyCL *proxy_cl_{nullptr};

    void GenerateObjects();
    void PivotCollision(Model3D *object);
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
