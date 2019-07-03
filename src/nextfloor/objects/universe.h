/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_UNIVERSE_H_
#define NEXTFLOOR_OBJECTS_UNIVERSE_H_

#include <memory>
#include <vector>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Universe
 *  @brief Define an Universe (with rooms inside), inherits Model3D abstract class
 */
class Universe : public Model {

public:

    Universe();

    Universe(Universe&&) = default;

    Universe& operator=(Universe&&) = default;

    Universe(const Universe&) = delete;

    Universe& operator=(const Universe&) = delete;

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

    /*
     *  Mutators
     */
    void toready() { ready_ = true; }

    /**
     *   Add Doors and Windows for Rooms object
     */
    void InitDoorsForRooms() noexcept;


private:

    /** turn to true after 10 firt frames */
    bool ready_{false};

    /** filtering only rooms with display state */
    std::vector<EngineObject*> display_rooms_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_UNIVERSE_H_

