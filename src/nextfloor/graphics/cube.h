/**
 *  @file cube.h
 *  @brief Cube class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRAPHICS_CUBE_H_
#define NEXTFLOOR_GRAPHICS_CUBE_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "nextfloor/graphics/shape3d.h"


namespace nextfloor {

namespace graphics {

/**
 *  @class Cube
 *  @brief Define a 3d Cube Object, Inherits from Shape3D
 */
class Cube : public Shape3D {

public:

    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move);

    Cube(float scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer);

    Cube(Cube&&) = default;

    Cube& operator=(Cube&&) = default;

    Cube(const Cube&) = default;

    Cube& operator=(const Cube&) = default;

    virtual ~Cube() override = default;

    void Draw() noexcept override final;

private:

    /** The vertex buffer reference */
    GLuint vertexbuffer_;

    /** The texture buffer reference */
    GLuint texturebuffer_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_CUBE_H_
