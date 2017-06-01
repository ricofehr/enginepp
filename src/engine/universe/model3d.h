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

#include "engine/geometry/shape3d.h"
#include "engine/geometry/box.h"
#include "engine/physics/collision_engine.h"

namespace engine {
namespace universe {

/* Forward declaration of Camera class */
class Camera;

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

    Model3D();
    /* Default move and copy constructor / assignments */
    Model3D(Model3D&&) = default;
    Model3D& operator=(Model3D&&) = default;

    Model3D(const Model3D&) = default;
    Model3D& operator=(const Model3D&) = default;

    virtual ~Model3D();

    /* Draw functions */
    void PrepareDraw(Camera *cam);
    void Draw();

    /* Grid state functions */
    virtual std::vector<std::unique_ptr<Model3D>> ReinitGrid();
    std::unique_ptr<Model3D> TransfertObject(std::unique_ptr<Model3D> obj, bool force);
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects();

    /* Collision Function */
    void DetectCollision(std::vector<Model3D*> neighbors);

    /* Operators */
    friend bool operator==(const Model3D &o1, const Model3D &o2);
    friend bool operator!=(const Model3D &o1, const Model3D &o2);

    /* Accessors */
    int id() const { return id_; }
    float distance() const { return distance_; }
    int id_last_collision() const { return id_last_collision_; }
    Model3D *obstacle() { return obstacle_; }
    engine::geometry::Box border() const { return border_; }
    bool IsCrossed() const { return is_crossed_; }
    bool IsControlled() const { return is_controlled_; }
    int type() const { return type_; }
    std::vector<std::vector<int>> get_placements() const { return placements_; }
    virtual int countObjects() const { return objects_.size(); }
    inline virtual int countMovingObjects() const {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
            if (objects_[cnt]->IsMoved()) {
                *count_sum += 1;
            }
        }
        return count_sum.get_value();
    }
    bool IsFull() const { return nbobjects_ <= 0; }
    bool IsCamera() const { return type_ == kMODEL3D_CAMERA; }
    Model3D *get_camera()
    {
        if (objects_[0]->IsCamera()) {
            return objects_[0].get();
        }

        return nullptr;
    }
    std::vector<Model3D*> getObjects(int i, int j, int k) const { return grid_[i][j][k]; }

    /* Delegate Accessors */
    glm::vec3 location() const { return glm::vec3(border_.location()); }
    bool IsMoved() const { return border_.IsMoved(); }
    bool IsMovedX() const { return border_.IsMovedX(); }
    bool IsMovedY() const { return border_.IsMovedY(); }
    bool IsMovedZ() const { return border_.IsMovedZ(); }

    /* Mutators */
    void set_distance(float distance) { distance_ = distance; }
    void set_obstacle(Model3D *obstacle) { obstacle_ = obstacle; }
    void add_placement(int i, int j, int k) { placements_.push_back({i,j,k}); }
    void clear_placements() { placements_.clear(); }
    void set_camera(std::unique_ptr<Model3D> cam) { objects_.insert(objects_.begin(), std::move(cam)); }

    /* Delegate Mutators */
    inline void InverseMove() {
        border_.InverseMove();
        for (auto &e : elements_) {
            e->InverseMove();
        }
    }
    void lock() { object_mutex_.lock(); }
    void unlock() { object_mutex_.unlock(); }
    void MoveCamera();

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
    void DisplayGrid();
    void PivotCollision(Model3D *object, std::vector<Model3D*> neighbors);
    bool IsInside (glm::vec3 location_object) const;

    /* Compute neighbors */
    std::vector<Model3D*> GetOrderNeighbors(Model3D *r);
    std::map<int, Model3D*> GetNeighbors(Model3D *r, int level);

    std::vector<std::unique_ptr<engine::geometry::Shape3D>> elements_;
    engine::geometry::Box border_;
    std::vector<Model3D*> ***grid_{nullptr};
    std::vector<std::unique_ptr<Model3D>> objects_;
    std::vector<std::vector<int>> placements_;
    tbb::mutex object_mutex_;
    tbb::mutex grid_mutex_;
    Model3D *obstacle_;
    engine::physics::CollisionEngine *collision_engine_;

    float distance_;
    int id_;
    int id_last_collision_;
    int type_{10000};
    int nbobjects_{33};
    bool is_crossed_;
    bool is_controlled_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_MODEL3D_H_
