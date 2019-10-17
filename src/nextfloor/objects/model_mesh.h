/**
 *  @file model_mesh.h
 *  @brief ModelMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODELMESH_H_
#define NEXTFLOOR_OBJECTS_MODELMESH_H_

#include "nextfloor/objects/mesh.h"

#include <memory>
#include <tbb/mutex.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/grid_box.h"

namespace nextfloor {

namespace objects {

/**
 *  @class ModelMesh
 *  @brief Abstract class who defines generic 3d object and shall be inherited by all other 3d models
 */
class ModelMesh : public Mesh {

public:
    ~ModelMesh() override = default;

    friend bool operator==(const ModelMesh& o1, const ModelMesh& o2);
    friend bool operator!=(const ModelMesh& o1, const ModelMesh& o2);

    std::vector<std::pair<glm::mat4, std::string>> GetModelViewProjectionsAndTextureToDraw() const override;
    std::vector<Mesh*> GetMovingObjects() override;
    std::vector<Mesh*> FindCollisionNeighbors() const final;
    std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const final;
    bool IsNeighborEligibleForCollision(const Mesh& neighbor) const final;
    void MoveLocation() override;
    void UpdateGridPlacement() override;

    Mesh* AddIntoChild(std::unique_ptr<Mesh> mesh) final;
    bool IsInside(const Mesh& mesh) const final;
    bool IsInside(const glm::vec3& location) const final;
    Mesh* add_child(std::unique_ptr<Mesh> object) final;
    std::unique_ptr<Mesh> remove_child(Mesh* child) final;
    Mesh* TransfertChildToNeighbor(Mesh* child) final;

    void UpdateChildPlacement(Mesh* object) final;
    void AddMeshToGrid(Mesh* object) final;
    void RemoveItemsToGrid(Mesh* object) final;

    bool IsLastObstacle(Mesh* obstacle) const final;
    void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) override;

    bool IsFrontPositionFilled() const final;
    bool IsRightPositionFilled() const final;
    bool IsBackPositionFilled() const final;
    bool IsLeftPositionFilled() const final;
    bool IsBottomPositionFilled() const final;
    bool IsTopPositionFilled() const final;

    int id() const final { return id_; }
    glm::vec3 location() const final { return border_->location(); }
    glm::vec3 dimension() const final { return border_->dimension(); }
    float diagonal() const final { return border_->diagonal(); }
    Grid* grid() const { return grid_.get(); }
    bool IsCamera() const override { return false; }
    bool IsPlayer() const override { return false; }
    glm::vec3 movement() const final { return border_->movement(); }

    void set_parent(Mesh* parent) final { parent_ = parent; }
    void set_movement(const glm::vec3& movement) final
    {
        border_->set_movement(movement);
        for (auto& object : objects_) {
            object->set_movement(movement);
        }

        for (auto& polygon : polygons_) {
            polygon->set_movement(movement);
        }
    }

    void set_move_factor(float move_factor) final
    {
        border_->set_move_factor(move_factor);
        for (auto& object : objects_) {
            object->set_move_factor(move_factor);
        }

        for (auto& polygon : polygons_) {
            polygon->set_move_factor(move_factor);
        }
    }

    std::vector<glm::vec3> getCoordsModelMatrixComputed() const final
    {
        return border_->getCoordsModelMatrixComputed();
    }

    Border* border() const final { return border_.get(); }

    std::vector<glm::ivec3> coords() const final;
    bool hasChilds() const final { return objects_.size() != 0; }
    bool hasNoChilds() const final { return objects_.size() == 0; }

    std::vector<Mesh*> AllStubMeshs() final;

    std::vector<Mesh*> childs() const final
    {
        std::vector<Mesh*> ret_childs(0);
        for (auto& object : objects_) {
            ret_childs.push_back(object.get());
        }
        return ret_childs;
    }

    std::vector<Mesh*> descendants() const final
    {
        std::vector<Mesh*> ret_childs(0);
        for (auto& object : objects_) {
            if (object->hasNoChilds()) {
                ret_childs.push_back(object.get());
            }
            else {
                auto grant_childs = object->descendants();
                ret_childs.insert(ret_childs.end(), grant_childs.begin(), grant_childs.end());
            }
        }
        return ret_childs;
    }

    void InitChildsIntoGrid() override;
    void AddIntoAscendantGrid() final;

protected:
    ModelMesh();

    ModelMesh(ModelMesh&&) = delete;
    ModelMesh& operator=(ModelMesh&&) = delete;
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;

    bool IsMoved() const { return border_->IsMoved(); }
    void ResetObstacle();

    Mesh* parent_{nullptr};
    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Grid> grid_{nullptr};
    std::vector<std::unique_ptr<Mesh>> objects_;
    std::vector<std::unique_ptr<nextfloor::polygons::Polygon>> polygons_;
    std::unique_ptr<Border> border_{nullptr};

private:
    void InitCollisionEngine();
    void set_gridcoords(std::vector<GridBox*> coords_list) { coords_list_ = coords_list; }

    bool IsNeighborReachable(const Mesh& neighbor) const;
    bool IsInDirection(const Mesh& target) const;

    void LogCollision(const Mesh& obstacle, float obstacle_distance);

    int id_{0};

    Mesh* obstacle_{nullptr};

    tbb::mutex mutex_;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MODELMESH_H_
