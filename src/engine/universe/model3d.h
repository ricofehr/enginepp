/*
* Model3D class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_MODEL3D_H_
#define ENGINE_UNIVERSE_MODEL3D_H_

#include <memory>
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
    static constexpr int kMODEL3D_ROOM = 0;
    static constexpr int kMODEL3D_CAMERA = 1;
    static constexpr int kMODEL3D_WALL = 2;
    static constexpr int kMODEL3D_WINDOW = 3;
    static constexpr int kMODEL3D_DOOR = 4;
    static constexpr int kMODEL3D_BRICK = 5;

    /* Grid Constants */
    static constexpr int kGRID_Y = 6;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT = 2.0f;

    Model3D();
    /* Default move and copy constructor / assignments */
    Model3D(Model3D&&) = default;
    Model3D& operator=(Model3D&&) = default;

    Model3D(const Model3D&) = default;
    Model3D& operator=(const Model3D&) = default;

    /* Default destructor */
    virtual ~Model3D() = default;

    /* Draw functions */
    void PrepareDraw(Camera *cam);
    void Draw();

    /* Grid state functions */
    virtual std::vector<std::unique_ptr<Model3D>> ReinitGrid();
    std::unique_ptr<Model3D> TransfertObject(std::unique_ptr<Model3D> obj, bool force);
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects();

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
    Camera * cam() const { return cam_; }
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

    void InitCollisionEngine();
    void DisplayGrid();
    bool IsInside (glm::vec3 location_object) const;

    std::vector<std::unique_ptr<engine::geometry::Shape3D>> elements_;
    engine::geometry::Box border_;
    std::vector<Model3D*> grid_[kGRID_Y][kGRID_X][kGRID_Z];
    std::vector<std::unique_ptr<Model3D>> objects_;
    std::vector<std::vector<int>> placements_;
    tbb::mutex object_mutex_;
    tbb::mutex grid_mutex_;
    Model3D *obstacle_;
    engine::physics::CollisionEngine *collision_engine_;
    Camera *cam_{nullptr};
    glm::vec4 location_;
    float distance_;
    int id_;
    int id_last_collision_;
    int type_;
    int nbobjects_{33};
    bool is_crossed_;
    bool is_controlled_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_MODEL3D_H_
