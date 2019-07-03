/**
 *  @file cube.h
 *  @brief Cube class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_CUBE_H_
#define NEXTFLOOR_POLYGONS_CUBE_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "nextfloor/polygons/polygon.h"


namespace nextfloor {

namespace polygons {

/**
 *  @class Cube
 *  @brief Define a 3d Cube Object, Inherits from Shape3D
 */
class Cube : public Polygon {

public:

    Cube(glm::vec3 scale, glm::vec4 location);

    Cube(float scale, glm::vec4 location, GLuint vertexbuffer, GLuint texturebuffer);

    Cube(glm::vec3 scale, glm::vec4 location, GLuint vertexbuffer, GLuint texturebuffer);

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

} // namespace polygons

} // namespace nextfloor

#endif // NEXTFLOOR_POLYGONS_CUBE_H_
