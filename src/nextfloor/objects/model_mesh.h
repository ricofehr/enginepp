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

    virtual Mesh* add_child(std::unique_ptr<Mesh> object) noexcept override final;
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) noexcept override;

    virtual bool IsLastObstacle(Mesh* obstacle) const noexcept override final;
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) noexcept override final;
    virtual void PrepareDraw() override {}

    virtual bool IsFrontPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsFrontPositionFilled();
    }

    virtual bool IsRightPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsRightPositionFilled();
    }

    virtual bool IsLeftPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsLeftPositionFilled();
    }

    virtual bool IsBackPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsBackPositionFilled();
    }

    virtual bool IsBottomPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsBottomPositionFilled();
    }

    virtual bool IsTopPositionFilled() const noexcept override
    {
        return coords_list_[0]->IsTopPositionFilled();
    }

    virtual bool IsCamera() const override { return type_ == kMODEL_CAMERA; }
    virtual void set_parent(Mesh* parent) override { parent_ = parent; }
    virtual int id() override { return id_; }
    virtual glm::vec3 location() const noexcept override { return border_->location(); }
    Grid* grid() const noexcept { return grid_.get(); }

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const noexcept override final
    {
        return border_->getCoordsModelMatrixComputed();
    }

    virtual Border* border() const noexcept override final
    {
        return border_.get();
    }

    virtual bool ready() const override { return ready_; }
    virtual void toready() override { ready_ = true; }


protected:

    ModelMesh();

    ModelMesh(ModelMesh&&) = default;
    ModelMesh& operator=(ModelMesh&&) = default;
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;

    void RemoveItemToGrid(Mesh* object) noexcept;

    virtual void lock() override final { mutex_.lock(); }
    virtual void unlock() override final { mutex_.unlock(); }

    Mesh* parent_{nullptr};
    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Grid> grid_{nullptr};
    std::vector<std::unique_ptr<Mesh>> objects_;
    std::vector<std::unique_ptr<Polygon>> polygons_;

    std::unique_ptr<Border> border_{nullptr};
    RendererEngine* renderer_{nullptr};

    int type_{10000};

private:

    void InitCollisionEngine();
    void AddItemToGrid(Mesh* object) noexcept;
    void set_gridcoords(std::vector<GridBox*> coords_list) { coords_list_ = coords_list; }

    int id_{0};

    Mesh* obstacle_{nullptr};

    /** turn to true after 10 firt frames */
    bool ready_{false};

    /** Mutex ensures thread safe instructions */
    tbb::mutex mutex_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_MODELMESH_H_
