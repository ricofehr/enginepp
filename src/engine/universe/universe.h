/*
 *  Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Define an Universe (with rooms inside), inherits Model3D abstract class
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

    /*
     *  Constructor
     */
    Universe();

    /* 
     *  Override the Model3D version
     *  Displays only childs (Rooms) filtered by clipping level
     */
    void Draw() noexcept override final;

    /*
     *  Accessors
     */
    const bool ready() const { return ready_; }

    /*
     *  Return number of Rooms follow displayed state or not
     */
    inline const int countRooms(bool display) const {
        if (display) {
            return display_rooms_.size();
        }
        return countChilds();
    }

    /*
     *  Return number of Obects inside Rooms follow displayed state or not
     */
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

    /*
     *  Return number of Moving Obects inside Rooms follow displayed state or not
     */
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

    /*
     *  Mutators
     */
    void toready() { ready_ = true; }

    /*
     *   Override Model3D Reinitgrid function
     *   Add Doors and Windows for Rooms object
     */
    void InitDoorsForRooms() noexcept;

private:

    /*
     *  Constants For Grid Settings
     */
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Y = 4;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_X = 16.0f;
    static constexpr float kGRID_UNIT_Y = 12.0f;
    static constexpr float kGRID_UNIT_Z = 16.0f;

    /*
     *  Universe Attributes
     *      ready_: turn to true after 10 firt frames
     *      display_rooms_: filtering only rooms with display state
     */
    bool ready_{false};
    std::vector<Model3D*> display_rooms_;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_UNIVERSE_H_

