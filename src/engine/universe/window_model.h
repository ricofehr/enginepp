/*
* WindowModel class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_WINDOWMODEL_H_
#define ENGINE_UNIVERSE_WINDOWMODEL_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"
#include "engine/geometry/quad.h"

namespace engine {
namespace universe {

/* Create a WindowModel model */
class WindowModel : public Model3D {

public:
    WindowModel();
    WindowModel(int face, float scale, glm::vec4 location);
    ~WindowModel() {}

private:
    int face_;

};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_WINDOWMODEL_H_

