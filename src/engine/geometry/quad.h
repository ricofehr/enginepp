/*
* Quad class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GEOMETRY_QUAD_H_
#define ENGINE_GEOMETRY_QUAD_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "engine/geometry/shape3d.h"

namespace engine {
namespace geometry {

/* Quad 3d object, inherit Shape3D */
class Quad : public Shape3D {

public:
    Quad(int face, float scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, float scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /* Default move and copy constructor / operator */
    Quad(Quad&&) = default;
    Quad& operator=(Quad&&) = default;

    Quad(const Quad&) = default;
    Quad& operator=(const Quad&) = default;

    /* Default destructor */
    ~Quad() override = default;

    void Draw() override;

    int face() const { return face_; }

private:
    int face_; /* 6 type of quad: back, front, left, right, top, roof */
    GLuint vertexbuffer_;
    GLuint texturebuffer_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_GEOMETRY_QUAD_H_

