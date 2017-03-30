/*
* Universe class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_UNIVERSE_H_
#define ENGINE_UNIVERSE_UNIVERSE_H_

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

#include "engine/universe/room.h"
#include "engine/universe/camera.h"
#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace universe {

/* Define the global universe */
class Universe {

public:
    /* Constructor */
    Universe();

    /* Compute new hop */
    void NextHop();

    /* Accessors */
    const Camera *cam() const { return cam_; }
    const bool ready() const { return ready_; }

    inline int countRooms(bool display) const {
        if (display)
            return display_rooms_.size();
        return rooms_.size();
    }

    inline int countObjects(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += rooms_[cnt]->countObjects();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < rooms_.size(); cnt++) {
                *count_sum += rooms_[cnt]->countObjects();
            }
        }
        return count_sum.get_value();
    }

    inline int countMovingObjects(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += rooms_[cnt]->countMovingObjects();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < rooms_.size(); cnt++) {
                *count_sum += rooms_[cnt]->countMovingObjects();
            }
        }
        return count_sum.get_value();
    }

    /* Mutators */
    const void toready() { ready_ = true; }

private:
    /* Constants */
    static constexpr int kGRID_Y = 4;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_X = 16.0f;
    static constexpr float kGRID_UNIT_Y = 12.0f;
    static constexpr float kGRID_UNIT_Z = 16.0f;

    /* Init Universe */
    void InitRooms();
    void InitProxyParallell();
    void CreateGLBuffers();
    /* Grid compute & display */
    void ReinitGrid();
    void DisplayGrid();
    /* Generate room(s) function */
    void GenerateRooms();
    void GenerateRandomRoom();
    Room *GenerateRoom(glm::vec4 location);
    void GenerateWalls();
    /* Compute neighbors */
    std::vector<Room*> GetOrderNeighbors(Room *r);
    std::map<int, Room*> GetNeighbors(Room *r, int level);

    /* Universe attributes */
    Camera *cam_{nullptr};
    std::vector<Room*> grid_[kGRID_Y][kGRID_X][kGRID_Z];
    std::unique_ptr<engine::parallell::EngineParallell> proxy_parallell_;
    std::vector<std::unique_ptr<Room>> rooms_;
    bool ready_{false};
    std::vector<Room*> display_rooms_;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_UNIVERSE_H_

