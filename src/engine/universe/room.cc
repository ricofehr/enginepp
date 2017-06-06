/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

namespace engine {
namespace universe {

/* Constructors */
Room::Room()
     :Room(glm::vec4(1.0f)) {}

Room::Room(glm::vec4 location)
     :Room(location, std::vector<bool>(6, false), std::vector<bool>(6, false)){}

Room::Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows) {
    type_ = kMODEL3D_ROOM;
    doors_ = is_doors;
    windows_ = is_windows;

    /* Init Grid Settings */
    grid_x_ = kGRID_X;
    grid_y_ = kGRID_Y;
    grid_z_ = kGRID_Z;
    grid_unit_x_ = kGRID_UNIT_X;
    grid_unit_y_ = kGRID_UNIT_Y;
    grid_unit_z_ = kGRID_UNIT_Z;
    InitGrid();

    using engine::graphics::Border;
    border_ = Border(glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2), location);
}

}//namespace universe
}//namespace engine
