/*
 *  Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *  
 *  Define a Room (inside with walls and moving objects), inherits Model3D abstract class
 */

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>
#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {

namespace universe {

class Room : public Model3D {

public:

    /*
     *  Constructors 
     */
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows);

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

    /*
     *  Room attributes
     *      doors_: select room side with a Door
     *      windows_: select room side with a Window
     */
    std::vector<bool> doors_{false, false, false, false, false, false};
    std::vector<bool> windows_{false, false, false, false, false, false};
};

} // namespace universe

} // namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
