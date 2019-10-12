/**
 *  @file mesh.h
 *  @brief Mesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MESH_H_
#define NEXTFLOOR_OBJECTS_MESH_H_

#include <memory>
#include <vector>
#include <list>
#include <glm/glm.hpp>

#include "nextfloor/objects/polygon.h"
#include "nextfloor/objects/border.h"
#include "nextfloor/objects/camera.h"

namespace nextfloor {

namespace objects {

class Grid;

/**
 *  @class Mesh
 *  @brief Interface who defines generic 3d object operations and shall be inherited by all other 3d models
 */
class Mesh {

public:
    virtual ~Mesh() = default;

    virtual std::vector<Polygon*> GetPolygonsReadyToDraw(const Camera& active_camera) const = 0;
    virtual std::vector<Mesh*> GetMovingObjects() = 0;
    virtual std::vector<Mesh*> FindCollisionNeighbors() const = 0;
    virtual std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const = 0;
    virtual bool IsNeighborEligibleForCollision(const Mesh& neighbor) const = 0;
    virtual void Move() = 0;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const = 0;

    virtual Mesh* AddIntoChild(std::unique_ptr<Mesh> mesh) = 0;
    virtual bool IsInside(const Mesh& mesh) const = 0;
    virtual bool IsInside(const glm::vec3& location) const = 0;
    virtual Mesh* add_child(std::unique_ptr<Mesh> child) = 0;
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) = 0;
    virtual Mesh* TransfertChildToNeighbor(Mesh* child) = 0;
    virtual void UpdateChildPlacement(Mesh* object) = 0;
    virtual void AddMeshToGrid(Mesh* object) = 0;
    virtual void RemoveItemsToGrid(Mesh* object) = 0;
    virtual std::vector<Mesh*> childs() = 0;
    virtual bool hasNoChilds() const = 0;
    virtual std::vector<Mesh*> AllStubMeshs() = 0;
    virtual std::vector<Mesh*> descendants() const = 0;
    virtual std::vector<glm::ivec3> coords() const = 0;

    virtual bool IsLastObstacle(Mesh* obstacle) const = 0;
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) = 0;

    virtual void InitChildsIntoGrid() = 0;
    virtual void AddIntoAscendantGrid() = 0;

    virtual void PrepareDraw(const Camera& active_camera) = 0;

    virtual bool IsFrontPositionFilled() const = 0;
    virtual bool IsRightPositionFilled() const = 0;
    virtual bool IsBackPositionFilled() const = 0;
    virtual bool IsLeftPositionFilled() const = 0;
    virtual bool IsBottomPositionFilled() const = 0;
    virtual bool IsTopPositionFilled() const = 0;

    virtual int id() const = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 dimension() const = 0;
    virtual glm::vec3 movement() const = 0;
    virtual float diagonal() const = 0;
    virtual Border* border() const = 0;
    virtual Camera* camera() const = 0;
    virtual std::list<Camera*> all_cameras() const = 0;
    virtual bool IsCamera() const = 0;
    virtual bool IsPlayer() const = 0;

    virtual void set_parent(Mesh* parent) = 0;
    virtual void set_movement(const glm::vec3& movement) = 0;
    virtual void set_move_factor(float move_factor) = 0;
    virtual void set_camera(std::unique_ptr<Camera> camera) = 0;
    virtual void TransferCameraToOtherMesh(Mesh* other) = 0;

    virtual void lock() = 0;
    virtual void unlock() = 0;

    virtual bool ready() const = 0;
    virtual void toready() = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MESH_H_
