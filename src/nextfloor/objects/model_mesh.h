/**
 *  @file model_mesh.h
 *  @brief ModelMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODELMESH_H_
#define NEXTFLOOR_OBJECTS_MODELMESH_H_

#include "nextfloor/objects/mesh.h"

#include <vector>
#include <tbb/mutex.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/border.h"
#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/grid_box.h"
#include "nextfloor/objects/polygon.h"

namespace nextfloor {

namespace objects {

/**
 *  @class ModelMesh
 *  @brief Abstract class who defines generic 3d object and shall be inherited by all other 3d models
 */
class ModelMesh : public Mesh {

public:
    static constexpr int kMODEL_CAMERA = 2;

    ~ModelMesh() override = default;

    friend bool operator==(const ModelMesh& o1, const ModelMesh& o2);
    friend bool operator!=(const ModelMesh& o1, const ModelMesh& o2);

    std::vector<Polygon*> GetPolygonsReadyToDraw(const Camera& active_camera) const override;
    std::vector<Mesh*> GetMovingObjects() override;
    std::vector<Mesh*> FindCollisionNeighbors() const final;
    std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const final;
    bool IsNeighborEligibleForCollision(const Mesh& neighbor) const final;
    void Move() override;

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
    void PrepareDraw(const Camera& active_camera) override;

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
    Camera* camera() const final;
    std::list<Camera*> all_cameras() const final;
    bool IsCamera() const final { return camera_ != nullptr; }
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

    void set_camera(std::unique_ptr<Camera> camera) final { camera_ = std::move(camera); }
    void TransferCameraToOtherMesh(Mesh* other) final;

    std::vector<glm::vec3> getCoordsModelMatrixComputed() const final
    {
        return border_->getCoordsModelMatrixComputed();
    }

    Border* border() const final { return border_.get(); }

    std::vector<glm::ivec3> coords() const final;
    bool ready() const final { return ready_; }
    void toready() final { ready_ = true; }

    bool hasNoChilds() const final { return objects_.size() == 0; }

    std::vector<Mesh*> AllStubMeshs() final;

    std::vector<Mesh*> childs() final
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

    void lock() final { mutex_.lock(); }
    void unlock() final { mutex_.unlock(); }

    bool IsMoved() const { return border_->IsMoved(); }
    void ResetObstacle();

    Mesh* parent_{nullptr};
    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Grid> grid_{nullptr};
    std::vector<std::unique_ptr<Mesh>> objects_;
    std::vector<std::unique_ptr<Polygon>> polygons_;
    std::unique_ptr<Border> border_{nullptr};
    std::unique_ptr<Camera> camera_{nullptr};

private:
    void InitCollisionEngine();
    void set_gridcoords(std::vector<GridBox*> coords_list) { coords_list_ = coords_list; }

    bool IsNeighborReachable(const Mesh& neighbor) const;
    bool IsInDirection(const Mesh& target) const;

    void LogCollision(const Mesh& obstacle, float obstacle_distance);

    int id_{0};

    Mesh* obstacle_{nullptr};

    /** turn to true after 10 first frames */
    bool ready_{false};

    tbb::mutex mutex_;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MODELMESH_H_
