/**
 *  @file room.h
 *  @brief Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>
#include <glm/glm.hpp>

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
 *  @class Room
 *  @brief Define a Room (inside with walls and moving objects), inherits Model3D abstract class
 */
class Room : public Model3D {

public:

    /**
     *  Constructor
     */
    Room();

    /**
     *  Constructor
     *  @param location is the center point of the room
     */
    Room(glm::vec4 location);

    /**
     *  Constructor
     *  @param location is the center point of the room
     *  @param is_doors set sides which are a door
     *  @param is_windows set sides which are a window
     */
    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows);

    /**
     *  Default Move constructor
     */
    Room(Room&&) = default;

    /**
     *  Default Move assignment
     */
    Room& operator=(Room&&) = default;

    /**
     *  Copy constructor Deleted (Model3D Inherit)
     */
    Room(const Room&) = delete;

    /**
     *  Copy assignment Deleted (Model3D Inherit)
     */
    Room& operator=(const Room&) = delete;

    /**
     *  Default destructor
     */
    ~Room() override = default;

    /*
     *  Accessors
     */
    const std::vector<bool> doors() const { return doors_; }
    const std::vector<bool> windows() const { return windows_; }

    /*
     *  Mutators
     */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

private:

    /*
     *  Grid Constants For Romm Setting
     */
    static constexpr int kGRID_Y = 6;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT_Y = 2.0f;
    static constexpr float kGRID_UNIT_X = 2.0f;
    static constexpr float kGRID_UNIT_Z = 2.0f;

    /** select room side with a Door */
    std::vector<bool> doors_{false, false, false, false, false, false};

    /** select room side with a Window */
    std::vector<bool> windows_{false, false, false, false, false, false};
};

} // namespace universe

} // namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
