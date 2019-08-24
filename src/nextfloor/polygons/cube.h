/**
 *  @file cube.h
 *  @brief Cube class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_CUBE_H_
#define NEXTFLOOR_POLYGONS_CUBE_H_

#include "nextfloor/polygons/mesh_polygon.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace polygons {

/**
 *  @class Cube
 *  @brief Define a 3d Cube Object, Inherits from Shape3D
 */
class Cube : public MeshPolygon {

public:
    Cube(glm::vec3 location, glm::vec3 scale);
    Cube(glm::vec3 location, float scale);

    Cube(Cube&&) = default;
    Cube& operator=(Cube&&) = default;
    Cube(const Cube&) = default;
    Cube& operator=(const Cube&) = default;

    virtual ~Cube() override = default;

    void Draw(nextfloor::objects::RendererEngine* renderer) noexcept override final;
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_CUBE_H_
