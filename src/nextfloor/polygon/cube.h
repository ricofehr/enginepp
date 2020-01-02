/**
 *  @file cube.h
 *  @brief Cube class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGON_CUBE_H_
#define NEXTFLOOR_POLYGON_CUBE_H_

#include "nextfloor/polygon/mesh_polygon.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace polygon {

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
    static constexpr char kNoTexture[] = "";
};

}  // namespace polygon

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_CUBE_H_
