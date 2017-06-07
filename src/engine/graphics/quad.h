/*
 *   Quad class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *   Define a 3d Quad Object, Inherits from Shape3D.
 */

#ifndef ENGINE_GRAPHICS_QUAD_H_
#define ENGINE_GRAPHICS_QUAD_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "engine/graphics/shape3d.h"

namespace engine {

namespace graphics {

class Quad : public Shape3D {

public:

    /*
     *  Constructors
     */
    Quad(int face, float scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, float scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /*
     *  Default move constructor and assignment
     */
    Quad(Quad&&) = default;
    Quad& operator=(Quad&&) = default;

    /*
     *  Default copy constructor and assignment
     */
    Quad(const Quad&) = default;
    Quad& operator=(const Quad&) = default;

    /*
     *  Default destructor
     */
    ~Quad() override = default;

    /*
     *  Draw the model in GL Scene
     */
    void Draw() noexcept override final;

    /*
     *  Accessor: return the side number
     */
    int face() const { return face_; }

private:

    GLuint vertexbuffer_;
    GLuint texturebuffer_;
    int face_; /* 6 types of quad: back, front, left, right, top, roof */
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_GRAPHICS_QUAD_H_

