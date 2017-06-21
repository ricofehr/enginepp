/**
 *  @file wall.h
 *  @brief Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_WALL_H_
#define NEXTFLOOR_UNIVERSE_WALL_H_

#include <glm/glm.hpp>

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
 *  @class Wall
 *  @brief Wall 3d model, inherits Model3D abstract class
 */
class Wall : public Model3D {

public:

    /*
     *  Side texture constants
     */
    static constexpr int kTEXTURE_TOP = 0;
    static constexpr int kTEXTURE_WALL = 1;
    static constexpr int kTEXTURE_FLOOR = 2;

    /*
     *  Wall Side constants
     */
    static constexpr int kWALL_FRONT = 0;
    static constexpr int kWALL_RIGHT = 1;
    static constexpr int kWALL_BACK = 2;
    static constexpr int kWALL_LEFT = 3;
    static constexpr int kWALL_BOTTOM = 4;
    static constexpr int kWALL_TOP = 5;

    /**
     *  Constructor
     */
    Wall();

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the brick
     *  @param face is the orientation number of the wall: kROOF, kFLOOR, kLEFT, kRIGHT, kFRONT, kBACK
     */
    Wall(glm::vec3 scale, glm::vec4 location, int face);

    /**
     *  Default move constructor
     */
    Wall(Wall&&) = default;

    /**
     *  Default move assignment
     */
    Wall& operator=(Wall&&) = default;

    /**
     *  Copy constructor Deleted (Model3D Inherit)
     */
    Wall(const Wall&) = delete;

    /**
     *  Copy assignment Deleted (Model3D Inherit)
     */
    Wall& operator=(const Wall&) = delete;

    /**
     *  Default destructor
     */
    ~Wall() override = default;

    /**
     *  Fill vertex and texture into GL Buffers
     *  Needs a sole execution for all Walls object.
     */
    static void CreateBuffers();
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_WALL_H_
