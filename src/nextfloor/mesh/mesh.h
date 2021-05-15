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
#include <tbb/mutex.h>

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

    friend bool operator==(const Mesh& o1, const Mesh& o2)
    {
        return o1.id_ == o2.id_;
    }

    friend bool operator!=(const Mesh& o1, const Mesh& o2)
    {
        return o1.id_ != o2.id_;
    }

    /* Dynamic methods - overrided by DynamicMesh */
    virtual std::vector<Mesh*> FindCollisionNeighbors() const { return std::vector<Mesh*>(0); }
    virtual bool IsNeighborEligibleForCollision(const Mesh& neighbor) const { return false; }
    virtual void MoveLocation() {}
    virtual glm::vec3 movement() const { return glm::vec3(0.0f); }
    virtual void set_distance_factor(float distance_factor) {}
    virtual void set_move_factor(glm::vec3 move_factor) {}
    virtual void set_movement(const glm::vec3& movement) {}

    /* Placement methods - overrided by PlacementMesh */
    virtual std::vector<Mesh*> GetMovingObjects();
    virtual bool IsLastObstacle(Mesh* obstacle) const { return false; }
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float distance_factor, glm::vec3 move_factor) {}

    glm::vec3 location() const;
    glm::vec3 dimension() const;
    float diagonal() const;
    Border* border() const;

    void set_gridcoords(std::vector<GridBox*> coords_list);
    void ClearCoords();
    std::vector<glm::ivec3> coords() const;
    void delete_gridcoord(GridBox* grid_box);

    /* Draw methods - overrided by DrawningMesh */
    virtual std::vector<std::pair<glm::mat4, std::string>> GetModelViewProjectionsAndTextureToDraw() const
    {
        return std::vector<std::pair<glm::mat4, std::string>>(0);
    }
    virtual void PrepareDraw(const glm::mat4& view_projection_matrix) {}

    /* Layout methodsi - overrided by ground objects */
    virtual bool hasLayout() const { return false; }
    virtual Mesh* UpdateChildPlacement(nextfloor::mesh::Mesh* child) { return nullptr; }
    virtual std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const { return std::vector<Mesh*>(0); }
    virtual Mesh* AddIntoChild(std::unique_ptr<nextfloor::mesh::Mesh> mesh) { return nullptr; }
    virtual bool IsInside(const Mesh& mesh) const { return false; }
    virtual bool IsFrontPositionFilled() const { return true; }
    virtual bool IsRightPositionFilled() const { return true; }
    virtual bool IsBackPositionFilled() const { return true; }
    virtual bool IsLeftPositionFilled() const { return true; }
    virtual bool IsBottomPositionFilled() const { return true; }
    virtual bool IsTopPositionFilled() const { return true; }

    /* Composite Object methods - Overrided by CompositeMesh */
    virtual int id() const { return id_; }
    virtual Mesh* add_child(std::unique_ptr<Mesh> child) { return nullptr; }
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) { return nullptr; }
    virtual std::vector<Mesh*> childs() const { return std::vector<Mesh*>(0); }
    virtual bool hasChilds() const { return false; }
    virtual bool hasNoChilds() const { return true; }
    virtual std::vector<Mesh*> leafs();
    virtual void set_parent(Mesh* parent) { parent_ = parent; }

    /* Other */
    virtual bool IsCamera() const { return false; }
    virtual bool IsPlayer() const { return false; }
    virtual std::string class_name() const { return std::string("Mesh"); }

protected:
    Mesh();

    Mesh* parent_{nullptr};
    int id_{0};
    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Border> border_{nullptr};
    Mesh* obstacle_{nullptr};
    tbb::mutex mutex_;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_MESH_H_
