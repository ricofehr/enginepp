/**
 *  @file model.h
 *  @brief Model class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODEL_H_
#define NEXTFLOOR_OBJECTS_MODEL_H_

#include "nextfloor/objects/engine_object.h"

#include <vector>
#include <tbb/mutex.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_border.h"
#include "nextfloor/objects/engine_grid.h"
#include "nextfloor/objects/engine_grid_box.h"
#include "nextfloor/objects/engine_renderer.h"
#include "nextfloor/objects/engine_polygon.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Model
 *  @brief Abstract class who defines generic 3d object and shall be inherited by all other 3d models
 */
class Model : public EngineObject {

public:

    static constexpr int kMODEL_CAMERA = 2;

    virtual ~Model() override = default;

    friend bool operator==(const Model& o1, const Model& o2);
    friend bool operator!=(const Model& o1, const Model& o2);

    virtual void Draw() noexcept override;

    virtual EngineObject* add_child(std::unique_ptr<EngineObject> object) noexcept override final;
    virtual std::unique_ptr<EngineObject> remove_child(EngineObject* child) noexcept override;

    virtual bool IsLastObstacle(EngineObject* obstacle) const noexcept override final;
    virtual void UpdateObstacleIfNearer(EngineObject* obstacle, float obstacle_distance) noexcept override final;
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
    virtual void set_parent(EngineObject* parent) override { parent_ = parent; }
    virtual int id() override { return id_; }
    virtual glm::vec3 location() const noexcept override { return border_->location(); }
    EngineGrid* grid() const noexcept { return grid_.get(); }

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const noexcept override final
    {
        return border_->getCoordsModelMatrixComputed();
    }

    virtual EngineBorder* border() const noexcept override final
    {
        return border_.get();
    }

    virtual bool ready() const override { return ready_; }
    virtual void toready() override { ready_ = true; }


protected:

    Model();

    Model(Model&&) = default;
    Model& operator=(Model&&) = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void RemoveItemToGrid(EngineObject* object) noexcept;

    virtual void lock() override final { mutex_.lock(); }
    virtual void unlock() override final { mutex_.unlock(); }

    EngineObject* parent_{nullptr};
    std::vector<EngineGridBox*> coords_list_;
    std::unique_ptr<EngineGrid> grid_{nullptr};
    std::vector<std::unique_ptr<EngineObject>> objects_;
    std::vector<std::unique_ptr<EnginePolygon>> meshes_;

    std::unique_ptr<EngineBorder> border_{nullptr};
    EngineRenderer* renderer_{nullptr};

    int type_{10000};

private:

    void InitCollisionEngine();
    void AddItemToGrid(EngineObject* object) noexcept;
    void set_gridcoords(std::vector<EngineGridBox*> coords_list) { coords_list_ = coords_list; }

    int id_{0};

    EngineObject* obstacle_{nullptr};

    /** turn to true after 10 firt frames */
    bool ready_{false};

    /** Mutex ensures thread safe instructions */
    tbb::mutex mutex_;
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_MODEL_H_
