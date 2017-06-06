/*
* Quad class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GRAPHICS_QUAD_H_
#define ENGINE_GRAPHICS_QUAD_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "engine/graphics/shape3d.h"

namespace engine {
namespace graphics {

/* Quad 3d object, inherit Shape3D */
class Quad : public Shape3D {

public:

    Quad(int face, float scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, float scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /* Default move and copy constructor / operator */
    Quad(Quad&&) = default;
    Quad& operator=(Quad&&) = default;

    Quad(const Quad&) = default;
    Quad& operator=(const Quad&) = default;

    /* Default destructor */
    ~Quad() override = default;

    void Draw() noexcept override final;

    int face() const { return face_; }

private:

    GLuint vertexbuffer_;
    GLuint texturebuffer_;
    int face_; /* 6 type of quad: back, front, left, right, top, roof */
};

}//namespace graphics
}//namespace engine

#endif //ENGINE_GRAPHICS_QUAD_H_

