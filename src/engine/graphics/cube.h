/*
 *   Cube class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *   Define a 3d Cube Object, Inherits from Shape3D.
 */

#ifndef ENGINE_GRAPHICS_CUBE_H_
#define ENGINE_GRAPHICS_CUBE_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "engine/graphics/shape3d.h"

namespace engine {

namespace graphics {

class Cube : public Shape3D {

public:

    /*
     *  Constructors
     */
    Cube(float scale, glm::vec4 location);
    Cube(float scale, glm::vec4 location, glm::vec4 move);
    Cube(glm::vec3 scale, glm::vec4 location);
    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move);
    Cube(float scale, glm::vec4 location, GLuint vertexbuffer, GLuint texturebuffer);
    Cube(float scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);
    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /*
     *  Default move constructor and assignment
     */
    Cube(Cube&&) = default;
    Cube& operator=(Cube&&) = default;

    /*
     *  Default copy constructor and assignment
     */
    Cube(const Cube&) = default;
    Cube& operator=(const Cube&) = default;

    /*
     *  Default destructor 
     */
    virtual ~Cube() override = default;

    /*
     *  Draw the model in GL Scene
     */
    void Draw() noexcept override final;

private:

    GLuint vertexbuffer_;
    GLuint texturebuffer_;
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_GRAPHICS_CUBE_H_
