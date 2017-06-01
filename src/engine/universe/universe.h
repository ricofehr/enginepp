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
#include "engine/physics/collision_engine.h"

namespace engine {
namespace universe {

/* Define the global universe */
class Universe : public Model3D {

public:
    /* Constructor */
    Universe();

    /* Compute new hop */
    void Draw();

    /* Accessors */
    const bool ready() const { return ready_; }

    inline int countRooms(bool display) const {
        if (display) {
            return display_rooms_.size();
        }
        return objects_.size();
    }

    inline int countObjects(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += display_rooms_[cnt]->countObjects();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
                *count_sum += objects_[cnt]->countObjects();
            }
        }
        return count_sum.get_value();
    }

    inline int countMovingObjects(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += display_rooms_[cnt]->countMovingObjects();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
                *count_sum += objects_[cnt]->countMovingObjects();
            }
        }
        return count_sum.get_value();
    }

    /* Mutators */
    const void toready() { ready_ = true; }

    /* Grid compute & display */
    std::vector<std::unique_ptr<Model3D>> ReinitGrid() override final;

private:

    /* Constants For Grid Settings */
    static constexpr int kGRID_Y = 4;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_X = 16.0f;
    static constexpr float kGRID_UNIT_Y = 12.0f;
    static constexpr float kGRID_UNIT_Z = 16.0f;

    /* Init Universe */
    void InitRooms();
    void CreateGLBuffers();

    /* Generate room(s) function */
    void GenerateRooms();
    void GenerateRandomRoom();
    Room *GenerateRoom(glm::vec4 location);
    void GenerateWalls();

    bool ready_{false};
    std::vector<Room*> display_rooms_;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_UNIVERSE_H_

