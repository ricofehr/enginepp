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
    Cube(const glm::vec3& location, const glm::vec3& scale);
    Cube(const glm::vec3& location, const glm::vec3& scale, const std::string& texture);
    ~Cube() final = default;

private:
    static constexpr char kNOTEXTURE[] = "";
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_CUBE_H_
