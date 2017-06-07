/*
 *   Model3D class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *   Abstract class who defines generic 3d model which shall be inherited by all other 3d models
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

#include "engine/graphics/shape3d.h"
#include "engine/graphics/border.h"
#include "engine/physics/collision_engine.h"

namespace engine {

namespace universe {

class Model3D {

public:

    /*
     *  Model Type Constants
     */
    static constexpr int kMODEL3D_UNIVERSE = 0;
    static constexpr int kMODEL3D_ROOM = 1;
    static constexpr int kMODEL3D_CAMERA = 2;
    static constexpr int kMODEL3D_WALL = 3;
    static constexpr int kMODEL3D_WINDOW = 4;
    static constexpr int kMODEL3D_DOOR = 5;
    static constexpr int kMODEL3D_BRICK = 6;

    /*
     *  Model Single Side Constants
     */
    static constexpr int kLEFT = 0;
    static constexpr int kRIGHT = 1;
    static constexpr int kFLOOR = 2;
    static constexpr int kROOF = 3;
    static constexpr int kFRONT = 4;
    static constexpr int kBACK = 5;

    /*
     *  Model Composed Side Constants
     */
    static constexpr int kLEFT_FLOOR = 6;
    static constexpr int kLEFT_ROOF = 7;
    static constexpr int kLEFT_FRONT = 8;
    static constexpr int kLEFT_FRONT_FLOOR = 9;
    static constexpr int kLEFT_FRONT_ROOF = 10;
    static constexpr int kLEFT_BACK = 11;
    static constexpr int kLEFT_BACK_FLOOR = 12;
    static constexpr int kLEFT_BACK_ROOF = 13;

    static constexpr int kRIGHT_FLOOR = 14;
    static constexpr int kRIGHT_ROOF = 15;
    static constexpr int kRIGHT_FRONT = 16;
    static constexpr int kRIGHT_FRONT_FLOOR = 17;
    static constexpr int kRIGHT_FRONT_ROOF = 18;
    static constexpr int kRIGHT_BACK = 19;
    static constexpr int kRIGHT_BACK_FLOOR = 20;
    static constexpr int kRIGHT_BACK_ROOF = 21;

    static constexpr int kFRONT_FLOOR = 22;
    static constexpr int kFRONT_ROOF = 23;
    static constexpr int kBACK_FLOOR = 24;
    static constexpr int kBACK_ROOF = 25;

    /*
     *  Destructor
     *  Deallocates grid_ raw pointers array
     */
    virtual ~Model3D();

    /*
     *  Record HID events
     */
    virtual void RecordHID();

    /*
     *  Proceed to move Model (if movable)
     */
    virtual void Move() noexcept;

    /*
     *  Draw the model and models inside
     */
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

    /*
     *   Compute again all placement coordinates for childs into the Grid
     */
    virtual std::vector<std::unique_ptr<Model3D>> ReinitGrid() noexcept;

    /*
     *  Add a child Model3D to current Model3D parent
     *  Return child if this one is not inside current Model3D parent
     */
    std::unique_ptr<Model3D> TransfertObject(std::unique_ptr<Model3D> child, bool force) noexcept;

    /*
     *  Return a list of Childs who were into the Grid
     *  but now are outside of the current Model
     */
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects() noexcept;

    /*
     *  Display the Grid into stdout for Debug purpose
     */
    void DisplayGrid() noexcept;

    /*
     *  Collision Function
     *  Detect the nearest Collision between the Current 3d model
     *  And the "neighbors" array in input
     */
    void DetectCollision(std::vector<Model3D*> neighbors) noexcept;

    /*
     *  (In)Equality Operators 
     */
    friend bool operator==(const Model3D &o1, const Model3D &o2);
    friend bool operator!=(const Model3D &o1, const Model3D &o2);

    /*
     *  Accessors 
     */
    int id() const { return id_; }
    float distance() const { return distance_; }
    int id_last_collision() const { return id_last_collision_; }
    Model3D* obstacle() { return obstacle_; }
    engine::graphics::Border* border() const { return border_.get(); }
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

    /*
     *  Delegate Accessors 
     */
    glm::vec3 location() const { return glm::vec3(border_->location()); }
    bool IsMoved() const { return border_->IsMoved(); }
    bool IsMovedX() const { return border_->IsMovedX(); }
    bool IsMovedY() const { return border_->IsMovedY(); }
    bool IsMovedZ() const { return border_->IsMovedZ(); }

    /*
     *  Mutators 
     */
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
    void set_obstacle(Model3D* obstacle) { obstacle_ = obstacle; }
    void add_placement(int i, int j, int k) { placements_.push_back({i,j,k}); }
    void clear_placements() { placements_.clear(); }
    void set_camera(std::unique_ptr<Model3D> cam) { objects_.insert(objects_.begin(), std::move(cam)); }
    void reset_missobjects(int missobjects) { missobjects_ = 0; }
    void set_missobjects(int missobjects) { missobjects_ = missobjects; }
    void inc_missobjects(int missobjects) { missobjects_ += missobjects; }

    /*
     *  Delegate Mutators 
     */
    inline void InverseMove() {
        border_->InverseMove();
        for (auto &e : elements_) {
            e->InverseMove();
        }
    }

    /*
     *  Lock / Unlock mutex of current model
     */
    void lock() { object_mutex_.lock(); }
    void unlock() { object_mutex_.unlock(); }

protected:

    /*
     *  Constructors
     *  Protected scope ensures Abstract Class Design
     */
    Model3D();

    /*
     *  Default move constructor and assignment
     *  Protected scope ensures Abstract Class Design
     */
    Model3D(Model3D&&) = default;
    Model3D& operator=(Model3D&&) = default;

    /*
     *  Delete copy constructor / assignment
     *  Because border_ is unique_ptr
     */
    Model3D(const Model3D&) = delete;
    Model3D& operator=(const Model3D&) = delete;

    /*
     *  Select the Collision Engine Algorithm
     */
    void InitCollisionEngine();

    /*
     *  Allocate grid array dynamically
     *  Uses of raw pointers for 3d arrays
     *  And performance
     */
    void InitGrid();

    /*
     *  Pivot around one child (parameter target) of the Current Model3D.
     *  And compute collisions between this child and the other childs of 
     *  current Model3D and his nearest "brothers" (neighbors parameter).
     */
    void PivotCollision(Model3D* target, std::vector<Model3D*> neighbors) noexcept;

    /*
     *  Check if location_object point is inside the current Model
     */
    bool IsInside (glm::vec3 location_object) const;

    /* 
     *  Return 6 side "brothers" (Model3D with same mother) of target if exists
     *  Indexed by side constants
     */
    std::vector<Model3D*> GetOrderNeighbors(Model3D* target) noexcept;

    /*
     *  Return a group of 3d models near target with a deeping clipping level
     */
    std::map<int, Model3D*> GetNeighbors(Model3D* target, int level) noexcept;

    /*
     *  Model3D Composite attributes
     *      elements_: 3d objects which composes the 3d model
     *      border_: the "box" which defines the border of the 3d model
     *      objects_: 3d models which are inside the current 3d model, as childs of this one
     */
    std::vector<std::unique_ptr<engine::graphics::Shape3D>> elements_;
    std::unique_ptr<engine::graphics::Border> border_{nullptr};
    std::vector<std::unique_ptr<Model3D>> objects_;

    /*
     *  Grid placements for the childs of the model
     */
    std::vector<Model3D*> ***grid_{nullptr};

    /*
     *  Placements coordinates into the grid
     *  of the mother model for current model
     */
    std::vector<std::vector<int>> placements_;

    /*
     *  If defined, obestacle_ is the model collision partner
     */
    Model3D* obstacle_{nullptr};

    /*
     *  Mutex ensures thread safe
     */
    tbb::mutex object_mutex_;

    /*
     *  Engine used for collision computes
     */
    engine::physics::CollisionEngine* collision_engine_{nullptr};

    /*
     *  Grid settings
     *      grid_unit_(x|y|z)_: size in gl pixels for one unity and for the 3 dimensions
     *      grid_(x|y|z)_: number of unities in the complete grid for the 3 dimensions
     *      
     *      And Finally
     *           grid_x * grid_unit_x =~ width of the Model
     *           grid_y * grid_unit_y =~ heigth of the Model
     *           grid_z * grid_unit_z =~ depth of the Model
     */
    float grid_unit_x_{0.0f};
    float grid_unit_y_{0.0f};
    float grid_unit_z_{0.0f};
    int grid_x_{0};
    int grid_y_{0};
    int grid_z_{0};

    /*
     *  Model3d Attributes
     */
    float distance_;
    int id_;
    int id_last_collision_;
    int type_{10000};
    int missobjects_{0};
    bool is_crossed_;
    bool is_controlled_;
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_UNIVERSE_MODEL3D_H_
