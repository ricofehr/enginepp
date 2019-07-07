/**
 *  @file wallbrick.h
 *  @brief WallBrick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WALL_H_
#define NEXTFLOOR_OBJECTS_WALL_H_

#include "nextfloor/objects/model.h"

#include <glm/glm.hpp>
#include <string>


namespace nextfloor {

namespace objects {

/**
 *  @class WallBrick
 *  @brief WallBrick 3d model
 */
class WallBrick : public Model {

public:

    static constexpr char kWALL_TEXTURE[] = "assets/wall.png";
    static constexpr char kROOF_TEXTURE[] = "assets/sky.png";
    static constexpr char kFLOOR_TEXTURE[] = "assets/floor.png";

    WallBrick(glm::vec3 location, glm::vec3 scale, std::string texture);

    WallBrick(WallBrick&&) = default;
    WallBrick& operator=(WallBrick&&) = default;

    WallBrick(const WallBrick&) = delete;
    WallBrick& operator=(const WallBrick&) = delete;

    ~WallBrick() override = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_WALLBRICK_H_
