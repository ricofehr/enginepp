/**
 *  @file model.h
 *  @brief Model class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODEL_H_
#define NEXTFLOOR_OBJECTS_MODEL_H_

#include "nextfloor/objects/engine_object.h"

#include <memory>
#include <vector>
#include <map>
#include <tbb/mutex.h>
#include <tbb/tbb.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_border.h"
#include "nextfloor/objects/engine_grid.h"
#include "nextfloor/objects/engine_renderer.h"
#include "nextfloor/polygons/polygon.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Model
 *  @brief Abstract class who defines generic 3d object and shall be inherited by all other 3d models
 */
class Model : public EngineObject {

public:

    /*
     *  Model Type Constants
     */
    static constexpr int kMODEL_UNIVERSE = 0;
    static constexpr int kMODEL_ROOM = 1;
    static constexpr int kMODEL_CAMERA = 2;
    static constexpr int kMODEL_WALL = 3;
    static constexpr int kMODEL_WINDOW = 4;
    static constexpr int kMODEL_DOOR = 5;
    static constexpr int kMODEL_BRICK = 6;

    /*
     *  Initial collision countdown value
     */
    static constexpr int kCOLLISION_COUNTDOWN = 4;


    Model(Model&&) = default;

    Model& operator=(Model&&) = default;

    /* Copy constructor Deleted : border_ is unique_ptr */
    Model(const Model&) = delete;

    /* Copy assignment Deleted :  border_ is unique_ptr */
    Model& operator=(const Model&) = delete;

    virtual ~Model() override = default;

    friend bool operator==(const Model& o1, const Model& o2);
    friend bool operator!=(const Model& o1, const Model& o2);

    virtual void InitBorder(std::unique_ptr<EngineBorder> border) noexcept override final;

    virtual void Draw() noexcept override;

    virtual EngineObject* add_child(std::unique_ptr<EngineObject> object) noexcept override final;
    virtual std::unique_ptr<EngineObject> remove_child(EngineObject* child) noexcept override;

    virtual void lock() override final { object_mutex_.lock(); }
    virtual void unlock() override final { object_mutex_.unlock(); }
    virtual int id() override { return id_; }

    virtual void set_parent(EngineObject* parent) override { parent_ = parent; }

    virtual bool IsUniverse() const override { return type_ == kMODEL_UNIVERSE; }
    virtual bool IsRoom() const override { return type_ == kMODEL_ROOM; }
    virtual bool IsWall() const override { return type_ == kMODEL_WALL; }
    virtual bool IsCamera() const override { return type_ == kMODEL_CAMERA; }

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

    virtual bool IsLastObstacle(EngineObject* obstacle) const noexcept override final;

    virtual void UpdateObstacleIfNearer(EngineObject* obstacle, float obstacle_distance) noexcept override final;

protected:

    Model();

    /** meshes which composes the current object */
    std::vector<std::unique_ptr<nextfloor::polygons::Polygon>> meshes_;

    /** childs of the current object */
    std::vector<std::unique_ptr<EngineObject>> objects_;

    int type_{10000};

    std::unique_ptr<EngineGrid> grid_{nullptr};

    /** Parent of the current 3d model */
    EngineObject* parent_{nullptr};

    std::unique_ptr<EngineBorder> border_{nullptr};

    std::unique_ptr<EngineRenderer> renderer_{nullptr};

private:

    void InitCollisionEngine();

    int id_{0};

    /** Mutex ensures thread safe instructions */
    tbb::mutex object_mutex_;

    EngineObject* obstacle_{nullptr};
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_MODEL_H_
