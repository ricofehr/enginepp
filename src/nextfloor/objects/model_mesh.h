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
#include "nextfloor/objects/renderer_engine.h"
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

    virtual ~ModelMesh() override = default;

    friend bool operator==(const ModelMesh& o1, const ModelMesh& o2);
    friend bool operator!=(const ModelMesh& o1, const ModelMesh& o2);

    virtual void Draw() noexcept override;
    virtual void DetectCollision() noexcept override;
    virtual std::vector<Mesh*> FindCollisionNeighborsOf(Mesh* target) const noexcept override;
    virtual void Move() noexcept override;

    virtual Mesh* AddIntoChild(std::unique_ptr<Mesh> mesh) noexcept override final;
    virtual bool IsInside(Mesh* mesh) noexcept override final;
    virtual Mesh* add_child(std::unique_ptr<Mesh> object) noexcept override final;
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) noexcept override;

    virtual void UpdateItemToGrid(Mesh* object) noexcept override final;
    virtual void AddItemToGrid(Mesh* object) noexcept override final;
    virtual void RemoveItemToGrid(Mesh* object) noexcept override final;

    virtual bool IsLastObstacle(Mesh* obstacle) const noexcept override final;
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) noexcept override final;
    virtual void PrepareDraw() override {}

    virtual bool IsFrontPositionFilled() const noexcept override;
    virtual bool IsRightPositionFilled() const noexcept override;
    virtual bool IsBackPositionFilled() const noexcept override;
    virtual bool IsLeftPositionFilled() const noexcept override;
    virtual bool IsBottomPositionFilled() const noexcept override;
    virtual bool IsTopPositionFilled() const noexcept override;

    virtual int id() override { return id_; }
    virtual glm::vec3 location() const noexcept override { return border_->location(); }
    virtual glm::vec3 dimension() const noexcept override { return border_->dimension(); }
    virtual float diagonal() const noexcept override final { return border_->diagonal(); }
    Grid* grid() const noexcept { return grid_.get(); }
    virtual Camera* camera() const noexcept override;
    virtual bool IsCamera() const override { return camera_ != nullptr; }
    virtual glm::vec3 movement() const noexcept override { return border_->movement(); }

    virtual void set_parent(Mesh* parent) override { parent_ = parent; }
    virtual void set_movement(glm::vec3 movement) override
    {
        border_->set_movement(movement);
        for (auto &object : objects_) {
            object->set_movement(movement);
        }

        for (auto &polygon : polygons_) {
            polygon->set_movement(movement);
        }
    }

    virtual void set_move_factor(float move_factor) noexcept override
    {
        border_->set_move_factor(move_factor);
        for (auto &object : objects_) {
            object->set_move_factor(move_factor);
        }

        for (auto &polygon : polygons_) {
            polygon->set_move_factor(move_factor);
        }
    }

    virtual void set_camera(std::unique_ptr<Camera> camera) override { camera_ = std::move(camera); }
    virtual void TransferCameraToOtherMesh(Mesh* other) override;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const noexcept override final
    {
        return border_->getCoordsModelMatrixComputed();
    }

    virtual Border* border() const noexcept override final
    {
        return border_.get();
    }

    virtual std::vector<glm::ivec3> coords() override;
    virtual bool ready() const override { return ready_; }
    virtual void toready() override { ready_ = true; }

    virtual bool hasNoChilds() const override { return objects_.size() == 0; }

    virtual std::vector<Mesh*> childs() noexcept override
    {
        std::vector<Mesh*> ret_childs(0);
        for (auto &object : objects_) {
            ret_childs.push_back(object.get());
        }
        return ret_childs;
    }

    virtual std::vector<Mesh*> descendants() const noexcept override
    {
        std::vector<Mesh*> ret_childs(0);
        for (auto &object : objects_) {
            if (object->hasNoChilds()) {
                ret_childs.push_back(object.get());
            } else {
                auto grant_childs = object->descendants();
                ret_childs.insert(ret_childs.end(), grant_childs.begin(), grant_childs.end());
            }
        }
        return ret_childs;
    }

protected:

    ModelMesh();

    ModelMesh(ModelMesh&&) = default;
    ModelMesh& operator=(ModelMesh&&) = default;
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;

    virtual void lock() override final { mutex_.lock(); }
    virtual void unlock() override final { mutex_.unlock(); }

    Mesh* parent_{nullptr};
    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Grid> grid_{nullptr};
    std::vector<std::unique_ptr<Mesh>> objects_;
    std::vector<std::unique_ptr<Polygon>> polygons_;
    std::unique_ptr<Border> border_{nullptr};
    std::unique_ptr<Camera> camera_{nullptr};

    RendererEngine* renderer_{nullptr};

private:

    void InitCollisionEngine();
    void set_gridcoords(std::vector<GridBox*> coords_list) { coords_list_ = coords_list; }

    void PivotCollision() noexcept;
    //std::vector<Mesh*> FindCollisionNeighbors() const noexcept;
    void LogCollision(Mesh* obstacle, float obstacle_distance);

    bool IsMoved() const { return border_->IsMoved(); }

    int id_{0};

    Mesh* obstacle_{nullptr};

    /** turn to true after 10 first frames */
    bool ready_{false};

    tbb::mutex mutex_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_MODELMESH_H_
