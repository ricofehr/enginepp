/*
* Model3D class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_MODEL3D_H_
#define ENGINE_UNIVERSE_MODEL3D_H_

#include <memory>
#include <vector>
#include <map>
#include <tbb/mutex.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <glm/glm.hpp>

#include "engine/geometry/shape3d.h"
#include "engine/geometry/box.h"
#include "engine/physics/collision_engine.h"

namespace engine {
namespace universe {

/* Define a 3d model */
class Model3D {

public:

    /* Object Type Constants */
    static constexpr int kMODEL3D_UNIVERSE = 0;
    static constexpr int kMODEL3D_ROOM = 1;
    static constexpr int kMODEL3D_CAMERA = 2;
    static constexpr int kMODEL3D_WALL = 3;
    static constexpr int kMODEL3D_WINDOW = 4;
    static constexpr int kMODEL3D_DOOR = 5;
    static constexpr int kMODEL3D_BRICK = 6;

    /* Border Side Constants */
    static constexpr int kFLOOR = 0;
    static constexpr int kROOF = 1;
    static constexpr int kLEFT = 2;
    static constexpr int kRIGHT = 3;
    static constexpr int kFRONT = 4;
    static constexpr int kBACK = 5;

    /* Constructors, Copy / Move assignments and Destructor */
    Model3D();
    Model3D(Model3D&&) = default;
    Model3D& operator=(Model3D&&) = default;

    Model3D(const Model3D&) = default;
    Model3D& operator=(const Model3D&) = default;

    virtual ~Model3D();

    /* Draw and Move functions */
    virtual void RecordHID();
    virtual void Move() noexcept;
    inline virtual void Draw() noexcept {
        /* Draw current object */
        for (auto &element : elements_) {
            element->Draw();
        }

        /* Draw associated objects */
        for (auto &object : objects_) {
            object->Draw();
        }
    }

    /* Grid placements functions */
    virtual std::vector<std::unique_ptr<Model3D>> ReinitGrid() noexcept;
    std::unique_ptr<Model3D> TransfertObject(std::unique_ptr<Model3D> obj, bool force) noexcept;
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects() noexcept;
    void DisplayGrid() noexcept;

    /* Collision Function */
    void DetectCollision(std::vector<Model3D*> neighbors) noexcept;

    /* Operators */
    friend bool operator==(const Model3D &o1, const Model3D &o2);
    friend bool operator!=(const Model3D &o1, const Model3D &o2);

    /* Accessors */
    int id() const { return id_; }
    float distance() const { return distance_; }
    int id_last_collision() const { return id_last_collision_; }
    Model3D* obstacle() { return obstacle_; }
    engine::geometry::Box border() const { return border_; }
    bool IsCrossed() const { return is_crossed_; }
    bool IsControlled() const { return is_controlled_; }
    int type() const { return type_; }
    std::vector<std::vector<int>> placements() const { return placements_; }
    virtual int countChilds() const { return objects_.size(); }
    inline virtual int countMovingChilds() const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
            if (objects_[cnt]->IsMoved()) {
                *count_sum += 1;
            }
        }
        return count_sum.get_value();
    }
    constexpr int gridx() const { return grid_x_; }
    constexpr int gridy() const { return grid_y_; }
    constexpr int gridz() const { return grid_z_; }
    constexpr float grid_unitx() const { return grid_unit_x_; }
    constexpr float grid_unity() const { return grid_unit_y_; }
    constexpr float grid_unitz() const { return grid_unit_z_; }
    bool IsEmplacementGridEmpty(int l, int m, int n) const { return grid_[l][m][n].size() == 0; }
    bool IsFull() const { return missobjects_ <= 0; }
    bool IsCamera() const { return type_ == kMODEL3D_CAMERA; }
    inline Model3D* get_camera()
    {
        if (countChilds() > 0 &&
            objects_[0]->IsCamera()) {
            return objects_[0].get();
        }

        return nullptr;
    }
    std::vector<Model3D*> getPlacementObjects(int i, int j, int k) const { return grid_[i][j][k]; }

    /* Delegate Accessors */
    glm::vec3 location() const { return glm::vec3(border_.location()); }
    bool IsMoved() const { return border_.IsMoved(); }
    bool IsMovedX() const { return border_.IsMovedX(); }
    bool IsMovedY() const { return border_.IsMovedY(); }
    bool IsMovedZ() const { return border_.IsMovedZ(); }

    /* Mutators */
    inline Model3D* add_object(std::unique_ptr<Model3D> obj) noexcept {
        auto obj_raw = obj.get();

        for (auto &p : obj->placements()) {
            grid_[p[0]][p[1]][p[2]].push_back(obj_raw);
        }

        /* Dont decrement if Wall or Camera */
        if (obj_raw->type() != kMODEL3D_WALL &&
            obj_raw->type() != kMODEL3D_CAMERA) {
            --missobjects_;
        }

        objects_.push_back(std::move(obj));

        return obj_raw;
    }
    void set_distance(float distance) { distance_ = distance; }
    void set_obstacle(Model3D *obstacle) { obstacle_ = obstacle; }
    void add_placement(int i, int j, int k) { placements_.push_back({i,j,k}); }
    void clear_placements() { placements_.clear(); }
    void set_camera(std::unique_ptr<Model3D> cam) { objects_.insert(objects_.begin(), std::move(cam)); }
    void reset_missobjects(int missobjects) { missobjects_ = 0; }
    void set_missobjects(int missobjects) { missobjects_ = missobjects; }
    void inc_missobjects(int missobjects) { missobjects_ += missobjects; }

    /* Delegate Mutators */
    inline void InverseMove() {
        border_.InverseMove();
        for (auto &e : elements_) {
            e->InverseMove();
        }
    }
    void lock() { object_mutex_.lock(); }
    void unlock() { object_mutex_.unlock(); }

protected:

    /* Grid settings */
    int grid_x_{0};
    int grid_y_{0};
    int grid_z_{0};
    float grid_unit_x_{0.0f};
    float grid_unit_y_{0.0f};
    float grid_unit_z_{0.0f};

    void InitCollisionEngine();
    void InitGrid();
    void PivotCollision(Model3D *object, std::vector<Model3D*> neighbors) noexcept;
    bool IsInside (glm::vec3 location_object) const;

    /* Compute neighbors */
    std::vector<Model3D*> GetOrderNeighbors(Model3D *r) noexcept;
    std::map<int, Model3D*> GetNeighbors(Model3D *r, int level) noexcept;

    /* Model3D attributes */
    std::vector<std::unique_ptr<engine::geometry::Shape3D>> elements_;
    engine::geometry::Box border_;
    std::vector<Model3D*> ***grid_{nullptr};
    std::vector<std::unique_ptr<Model3D>> objects_;
    std::vector<std::vector<int>> placements_;
    tbb::mutex object_mutex_;
    Model3D *obstacle_;
    engine::physics::CollisionEngine *collision_engine_;

    float distance_;
    int id_;
    int id_last_collision_;
    int type_{10000};
    int missobjects_{0};
    bool is_crossed_;
    bool is_controlled_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_MODEL3D_H_
