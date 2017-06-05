/*
* Universe class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_UNIVERSE_H_
#define ENGINE_UNIVERSE_UNIVERSE_H_

#include <memory>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

#include "engine/universe/model3d.h"

namespace engine {
namespace universe {

/* Define the global universe */
class Universe : public Model3D {

public:
    /* Constructor */
    Universe();

    /* Compute new hop */
    void Draw() noexcept override final;

    /* Accessors */
    const bool ready() const { return ready_; }

    inline const int countRooms(bool display) const {
        if (display) {
            return display_rooms_.size();
        }
        return countChilds();
    }

    inline const int countRoomsChilds(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += display_rooms_[cnt]->countChilds();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
                *count_sum += objects_[cnt]->countChilds();
            }
        }
        return count_sum.get_value();
    }

    inline const int countRoomsMovingChilds(bool display) const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        if (display) {
            cilk_for(auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
                *count_sum += display_rooms_[cnt]->countMovingChilds();
            }
        } else {
            cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
                *count_sum += objects_[cnt]->countMovingChilds();
            }
        }
        return count_sum.get_value();
    }

    /* Mutators */
    void toready() { ready_ = true; }

    /* Grid compute & display */
    std::vector<std::unique_ptr<Model3D>> ReinitGrid() noexcept override final;

private:

    /* Constants For Grid Settings */
    static constexpr int kGRID_Y = 4;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_X = 16.0f;
    static constexpr float kGRID_UNIT_Y = 12.0f;
    static constexpr float kGRID_UNIT_Z = 16.0f;

    bool ready_{false};
    std::vector<Model3D*> display_rooms_;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_UNIVERSE_H_

