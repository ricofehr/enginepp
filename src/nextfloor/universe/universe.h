/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_UNIVERSE_H_
#define NEXTFLOOR_UNIVERSE_UNIVERSE_H_

#include <memory>
#include <vector>

#include "nextfloor/universe/model3d.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::universe
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
    bool ready() const { return ready_; }

    /**
     *  Return count of childs (rooms)
     *  @param display for count all rooms (=false), or only displayed rooms (=true)
     *  @return the count of rooms
     */
    inline int countRooms(bool display) const {
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
    inline int countRoomsChilds(bool display) const {
        if (display) {
            return tbb::parallel_reduce(
                tbb::blocked_range<int>(0, display_rooms_.size()),
                0, 
                [&](const tbb::blocked_range<int>& r, int init)->int {
                    for (int a = r.begin(); a != r.end(); ++a) {
                        init += display_rooms_[a]->countChilds();
                    }
                    return init;
                },
                [](int x, int y)->int {
                    return x + y;
                }
            );
        } else {
            return tbb::parallel_reduce(
                tbb::blocked_range<int>(0, objects_.size()),
                0, 
                [&](const tbb::blocked_range<int>& r, int init)->int {
                    for (int a = r.begin(); a != r.end(); ++a) {
                        init += objects_[a]->countChilds();
                    }
                    return init;
                },
                [](int x, int y)->int {
                    return x + y;
                }
            );
        }
    }

    /**
     *  Return count of only moving Room childs (Universe, bricks, ...)
     *  @param display for count all moving (=false), or only displayed (=true)
     *  @return the count of moving objects inside rooms
     */
    inline int countRoomsMovingChilds(bool display) const {
        if (display) {
            return tbb::parallel_reduce(
                tbb::blocked_range<int>(0, display_rooms_.size()),
                0, 
                [&](const tbb::blocked_range<int>& r, int init)->int {
                    for (int a = r.begin(); a != r.end(); ++a) {
                        init += display_rooms_[a]->countMovingChilds();
                    }
                    return init;
                },
                [](int x, int y)->int {
                    return x + y;
                }
            );
        } else {
            return tbb::parallel_reduce(
                tbb::blocked_range<int>(0, objects_.size()),
                0, 
                [&](const tbb::blocked_range<int>& r, int init)->int {
                    for (int a = r.begin(); a != r.end(); ++a) {
                        init += objects_[a]->countMovingChilds();
                    }
                    return init;
                },
                [](int x, int y)->int {
                    return x + y;
                }
            );
        }
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

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_UNIVERSE_H_

