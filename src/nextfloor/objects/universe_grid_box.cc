/**
 *  @file universe_grid_box.cc
 *  @brief UniverseGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe_grid_box.h"

#include "nextfloor/objects/room.h"

namespace nextfloor {

namespace objects {


UniverseGridBox::UniverseGridBox(glm::vec3 coords, EngineGrid* owner)
:GridBox(coords, owner) { }

void UniverseGridBox::AddRoomDoor(int side)
{
    auto occupant = dynamic_cast<nextfloor::objects::Room*>(getFirstOccupant());

    if (occupant != nullptr)
    {
        occupant->addDoor(side);
    }
}

void UniverseGridBox::AddRoomWindow(int side)
{
    auto occupant = dynamic_cast<nextfloor::objects::Room*>(getFirstOccupant());

    if (occupant != nullptr)
    {
        occupant->addWindow(side);
    }
}

} // namespace objects

} // namespace nextfloor
