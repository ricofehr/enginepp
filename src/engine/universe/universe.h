/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef ENGINE_UNIVERSE_UNIVERSE_H_
#define ENGINE_UNIVERSE_UNIVERSE_H_

#include <memory>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

#include "engine/universe/model3d.h"

/**
 *  @namespace engine
 *  @brief Common parent namespace
 */
namespace engine {

/**
 *  @namespace engine::universe
 *  @brief World elements
 */
namespace universe {

/**
 *  @class Universe
 *  @brief Define an Universe (with rooms inside), inherits Model3D abstract class
 */
class Universe : public Model3D {

public:

    /*
     *  Constructor
     */
    Universe();

    /**
     *  Default Move constructor
     */
    Universe(Universe&&) = default;

    /**
     *  Default Move assignment
     */
    Universe& operator=(Universe&&) = default;

    /**
     *  Copy constructor Deleted (Model3D Inherit)
     */
    Universe(const Universe&) = delete;

    /**
     *  Copy assignment Deleted (Model3D Inherit)
     */
    Universe& operator=(const Universe&) = delete;

    /**
     *  Default destructor
     */
    ~Universe() override = default;

    /**
     *  Override the Model3D version
     *  Displays only childs (Rooms) filtered by clipping level
     */
    void Draw() noexcept override final;

    /*
     *  Accessors
     */
    const bool ready() const { return ready_; }

    /**
     *  Return count of childs (rooms)
     *  @param display for count all rooms (=false), or only displayed rooms (=true)
     *  @return the count of rooms
     */
    inline const int countRooms(bool display) const {
        if (display) {
            return display_rooms_.size();
        }
        return countChilds();
    }

    /**
     *  Return count of all Room childs (walls, bricks, ...)
     *  @param display for count all (=false), or only displayed (=true)
     *  @return the count of objects inside rooms
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

    /**
     *  Return count of only moving Room childs (Universe, bricks, ...)
     *  @param display for count all moving (=false), or only displayed (=true)
     *  @return the count of moving objects inside rooms
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

    /**
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

    /** turn to true after 10 firt frames */
    bool ready_{false};

    /** filtering only rooms with display state */
    std::vector<Model3D*> display_rooms_;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_UNIVERSE_H_

