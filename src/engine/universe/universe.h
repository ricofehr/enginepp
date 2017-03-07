/*
* Universe class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_UNIVERSE_H_
#define ENGINE_UNIVERSE_UNIVERSE_H_

#include <iostream>
#include <memory>
#include <vector>

#include "engine/universe/room.h"
#include "engine/universe/camera.h"
#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace universe {

/* Define the global universe */
class Universe {

public:

    Universe(int type_parallell);
    ~Universe();

    void NextHop();

    const Camera *cam() const { return cam_.get(); }

private:
    void InitRooms();
    void InitCamera();
    void InitProxyParallell(int type_parallell);

    std::unique_ptr<engine::parallell::EngineParallell> proxy_parallell_;
    std::unique_ptr<Camera> cam_;
    std::vector<std::unique_ptr<Room>> rooms_;
    Room *active_room_;

};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_UNIVERSE_H_

