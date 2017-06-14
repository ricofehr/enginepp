/**
 *   Cube class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
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

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     */
    Cube(float scale, glm::vec4 location);

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     */
    Cube(float scale, glm::vec4 location, glm::vec4 move);

    /**
     *  Constructor
     *  @param scale is the 3 axis scale factor of native coords
     *  @param location is the center point of the quad
     */
    Cube(glm::vec3 scale, glm::vec4 location);

    /**
     *  Constructor
     *  @param scale is the 3 axis scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     */
    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move);

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Cube(float scale, glm::vec4 location, GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Cube(float scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Constructor
     *  @param scale is the 3 axis scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Default Move constructor
     */
    Cube(Cube&&) = default;

    /**
     *  Default Move assignment
     */
    Cube& operator=(Cube&&) = default;

    /**
     *  Default Copy constructor
     */
    Cube(const Cube&) = default;

    /**
     *  Default Copy assignment
     */
    Cube& operator=(const Cube&) = default;

    /**
     *  Default destructor 
     */
    virtual ~Cube() override = default;

    /**
     *  Draw the model in GL Scene
     */
    void Draw() noexcept override final;

private:

    /** The vertex buffer reference */
    GLuint vertexbuffer_;

    /** The texture buffer reference */
    GLuint texturebuffer_;
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_GRAPHICS_CUBE_H_
