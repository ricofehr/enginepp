/**
 *  @file quad.h
 *  @brief Quad class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRAPHICS_QUAD_H_
#define NEXTFLOOR_GRAPHICS_QUAD_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "nextfloor/graphics/shape3d.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::graphics
 *  @brief Elementary 3d graphic classes
 */
namespace graphics {

/**
 *  @class Quad
 *  @brief Define a 3d Quad Object, Inherits from Shape3D
 */
class Quad : public Shape3D {

public:

    /**
     *  Constructor
     *  @param face is the 3d side number of quad
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Quad(int face, float scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Constructor
     *  @param face is the 3d side number of quad
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Quad(int face, float scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Constructor
     *  @param face is the 3d side number of quad
     *  @param scale is the 3 axis scale factor of native coords
     *  @param location is the center point of the quad
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Constructor
     *  @param face is the 3d side number of quad
     *  @param scale is the 3 axis scale factor of native coords
     *  @param location is the center point of the quad
     *  @param move is the translation vector
     *  @param vertexbuffer is the id of gl vertex buffer
     *  @param texturebuffer is the id of gl texture buffer
     */
    Quad(int face, glm::vec3 scale, glm::vec4 location,
         glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    /**
     *  Default Move constructor
     */
    Quad(Quad&&) = default;

    /**
     *  Default Move assignment
     */
    Quad& operator=(Quad&&) = default;

    /**
     *  Default Copy constructor
     */
    Quad(const Quad&) = default;

    /**
     *  Default Copy assignment
     */
    Quad& operator=(const Quad&) = default;

    /**
     *  Default destructor
     */
    ~Quad() override = default;

    /**
     *  Draw the model in GL Scene
     */
    void Draw() noexcept override final;

    /**
     *  Accessor: return the face attribute value
     *  @return the side number of the quad
     */
    int face() const { return face_; }

private:

    /** The vertex buffer reference */
    GLuint vertexbuffer_;

    /** The texture buffer reference */
    GLuint texturebuffer_;

    /** 6 types of quad: back, front, left, right, top, roof */
    int face_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_QUAD_H_

