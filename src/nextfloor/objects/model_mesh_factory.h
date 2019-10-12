/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_
#define NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_

#include "nextfloor/objects/mesh_factory.h"

#include "nextfloor/objects/polygon_factory.h"
#include "nextfloor/objects/grid_factory.h"
#include "nextfloor/objects/physic_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class MeshFactory
 *  @brief Concret factory for Universe Objects
 */
class ModelMeshFactory : public MeshFactory {

public:
    ModelMeshFactory(PolygonFactory* polygon_factory, GridFactory* grid_factory, PhysicFactory* physic_factory);
    ~ModelMeshFactory() final = default;

    std::unique_ptr<Mesh> MakeUniverse() const final;
    std::unique_ptr<Mesh> MakeRoom(const glm::vec3& location) const final;

    std::unique_ptr<Mesh> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeWallBrick(const glm::vec3& location,
                                        const glm::vec3& scale,
                                        const std::string& texture) const final;

    std::unique_ptr<Mesh> MakeRock(const glm::vec3& location, const glm::vec3& movement) const final;
    std::unique_ptr<Mesh> MakeLittleRock(const glm::vec3& location, const glm::vec3& movement) const final;

private:
    std::vector<std::unique_ptr<Mesh>> GenerateWallBricks(glm::vec3 firstpoint,
                                                          glm::vec3 lastpoint,
                                                          const glm::vec3& brick_dimension,
                                                          const std::string& texture) const;

    PolygonFactory* polygon_factory_;
    GridFactory* grid_factory_;
    PhysicFactory* physic_factory_;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_
