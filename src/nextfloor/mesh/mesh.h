/**
 *  @file mesh.h
 *  @brief Mesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_MESH_H_
#define NEXTFLOOR_MESH_MESH_H_

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <string>

#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace mesh {

class GridBox;

/**
 *  @class Mesh
 *  @brief Interface who defines generic 3d object operations and shall be inherited by all other 3d models
 */
class Mesh {

public:
    virtual ~Mesh() = default;

    /* Dynamic methods */
    virtual std::vector<Mesh*> FindCollisionNeighbors() const = 0;
    virtual bool IsNeighborEligibleForCollision(const Mesh& neighbor) const = 0;
    virtual void MoveLocation() = 0;
    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const = 0;
    virtual glm::vec3 movement() const = 0;
    virtual void set_distance_factor(float distance_factor) = 0;
    virtual void set_move_factor(glm::vec3 move_factor) = 0;
    virtual void set_movement(const glm::vec3& movement) = 0;

    /* Placement methods */
    virtual std::vector<Mesh*> GetMovingObjects() = 0;
    virtual bool IsLastObstacle(Mesh* obstacle) const = 0;
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float distance_factor, glm::vec3 move_factor) = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 dimension() const = 0;
    virtual Border* border() const = 0;
    virtual float diagonal() const = 0;
    virtual void ClearCoords() = 0;
    virtual void set_gridcoords(std::vector<GridBox*> coords_list) = 0;
    virtual void delete_gridcoord(GridBox* grid_box) = 0;
    virtual std::vector<glm::ivec3> coords() const = 0;

    /* Draw methods */
    virtual std::vector<std::pair<glm::mat4, std::string>> GetModelViewProjectionsAndTextureToDraw() const = 0;
    virtual void PrepareDraw(const glm::mat4& view_projection_matrix) = 0;

    /* Layout methods */
    virtual bool hasLayout() const = 0;
    virtual std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const = 0;
    virtual bool IsInside(const Mesh& mesh) const = 0;
    virtual bool IsInside(const glm::vec3& location) const = 0;
    virtual Mesh* UpdateChildPlacement(nextfloor::mesh::Mesh* child) = 0;
    virtual Mesh* AddIntoChild(std::unique_ptr<nextfloor::mesh::Mesh> mesh) = 0;
    virtual bool IsFrontPositionFilled() const = 0;
    virtual bool IsRightPositionFilled() const = 0;
    virtual bool IsBackPositionFilled() const = 0;
    virtual bool IsLeftPositionFilled() const = 0;
    virtual bool IsBottomPositionFilled() const = 0;
    virtual bool IsTopPositionFilled() const = 0;

    /* Composite Object methods */
    virtual int id() const = 0;
    virtual Mesh* add_child(std::unique_ptr<Mesh> child) = 0;
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) = 0;
    virtual std::vector<Mesh*> childs() const = 0;
    virtual void set_parent(Mesh* parent) = 0;
    virtual bool hasChilds() const = 0;
    virtual bool hasNoChilds() const = 0;
    virtual std::vector<Mesh*> AllStubMeshs() = 0;
    virtual std::vector<Mesh*> descendants() const = 0;

    /* Other */
    virtual bool IsCamera() const = 0;
    virtual bool IsPlayer() const = 0;
    virtual std::string class_name() = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_MESH_H_
