/*
* Door class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_DOOR_H_
#define ENGINE_UNIVERSE_DOOR_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"
#include "engine/geometry/quad.h"

namespace engine {
namespace universe {

/* Create a Door model */
class Door : public Model3D {

public:
    Door();
    Door(int face, float scale, glm::vec4 location);

    /* Default move and copy constructor / operator */
    Door(Door&&) = default;
    Door& operator=(Door&&) = default;

    Door(const Door&) = default;
    Door& operator=(const Door&) = default;

    /* Default destructor */
    ~Door() override = default;

private:
    int face_;

};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_DOOR_H_

