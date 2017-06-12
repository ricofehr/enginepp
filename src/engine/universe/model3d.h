/*
 *   Model3D class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *   Abstract class who defines generic 3d model which shall be inherited by all other 3d models
 */

#ifndef ENGINE_UNIVERSE_MODEL3D_H_
#define ENGINE_UNIVERSE_MODEL3D_H_

#include <iostream>
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
    static constexpr int kLEFT_FLOOR_FRONT = 7;
    static constexpr int kLEFT_FLOOR_BACK = 8;
    static constexpr int kLEFT_ROOF = 9;
    static constexpr int kLEFT_ROOF_FRONT = 10;
    static constexpr int kLEFT_ROOF_BACK = 11;
    static constexpr int kLEFT_FRONT = 12;
    static constexpr int kLEFT_BACK = 13;

    static constexpr int kRIGHT_FLOOR = 14;
    static constexpr int kRIGHT_FLOOR_FRONT = 15;
    static constexpr int kRIGHT_FLOOR_BACK = 16;
    static constexpr int kRIGHT_ROOF = 17;
    static constexpr int kRIGHT_ROOF_FRONT = 18;
    static constexpr int kRIGHT_ROOF_BACK = 19;
    static constexpr int kRIGHT_FRONT = 20;
    static constexpr int kRIGHT_BACK = 21;

    static constexpr int kFLOOR_FRONT = 22;
    static constexpr int kFLOOR_BACK = 23;

    static constexpr int kROOF_FRONT = 24;
    static constexpr int kROOF_BACK = 25;
    static constexpr int kSAME = 26;

    static constexpr int kSIDES = 27;

    /*
     *  Grid Square Use Constants
     */
    static constexpr int kGRID_UNKNOW = -1;
    static constexpr int kGRID_USED = 0;
    static constexpr int kGRID_EMPTY = 1;

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
     *  Proceed to move current object
     */
    virtual void Move() noexcept;

    /*
     *  Draw the object and his childs
     */
    inline virtual void Draw() noexcept
    {
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
     *  Return a list of Childs who were into the Grid
     *  but now are outside of the current Model
     */
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects() noexcept;

    /*
     *  Display the Grid into stdout for Debug purpose
     */
    void DisplayGrid() const noexcept;

    /*
     *  Detect collision between childs of the current object
     */
    void DetectCollisionBetweenChilds() noexcept;

    /*
     *  Compute neighbors array with a deeping clipping level
     */
    void ComputeNeighbors() noexcept;

    /*
     *  Return a list of all neighbors of current object
     */
    std::vector<Model3D*> GetAllNeighbors() const noexcept;

    /*
     *  Return a list of neighbors in respect with the clipping constraint
     *  All objects too far or hidden by current view are not included.
     *      level: clipping level (1 -> high clipping, 2 -> low clipping)
     */
    std::vector<Model3D*> GetClippingNeighbors(int level) const noexcept;

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
    Model3D* parent() { return parent_; }
    engine::graphics::Border* border() const { return border_.get(); }
    bool IsCrossed() const { return is_crossed_; }
    bool IsControlled() const { return is_controlled_; }
    int type() const { return type_; }
    std::vector<std::vector<int>> placements() const { return placements_; }
    virtual int countChilds() const { return objects_.size(); }
    constexpr int gridx() const { return grid_x_; }
    constexpr int gridy() const { return grid_y_; }
    constexpr int gridz() const { return grid_z_; }
    constexpr float grid_unitx() const { return grid_unit_x_; }
    constexpr float grid_unity() const { return grid_unit_y_; }
    constexpr float grid_unitz() const { return grid_unit_z_; }
    int IsPositionInTheGridEmpty(int l, int m, int n) const noexcept;
    inline bool IsFull() const { return missobjects_ <= 0 || objects_.size() >= grid_x_ * grid_y_ * grid_z_; }
    bool IsCamera() const { return type_ == kMODEL3D_CAMERA; }

    /*
     *  Return Camera object if exists into childs array
     */
    inline Model3D* get_camera() const noexcept
    {
        if (countChilds() > 0 &&
            objects_[0]->IsCamera()) {
            return objects_[0].get();
        }

        return nullptr;
    }

    /*
     *  Return number of Moving Object into childs array
     */
    inline virtual int countMovingChilds() const
    {
        cilk::reducer<cilk::op_add<int>> count_sum(0);
        cilk_for(auto cnt = 0; cnt < objects_.size(); cnt++) {
            if (objects_[cnt]->IsMoved()) {
                *count_sum += 1;
            }
        }
        return count_sum.get_value();
    }

    /*
     *  Delegate Accessors 
     */
    glm::vec3 location() const { return glm::vec3(border_->location()); }
    bool IsMoved() const { return border_->IsMoved(); }
    int IsMovedX() const { return border_->IsMovedX(); }
    int IsMovedY() const { return border_->IsMovedY(); }
    int IsMovedZ() const { return border_->IsMovedZ(); }

    /*
     *  Mutators 
     */
    void set_distance(float distance) { distance_ = distance; }
    void set_obstacle(Model3D* obstacle) { obstacle_ = obstacle; }
    void reset_missobjects(int missobjects) { missobjects_ = 0; }
    void set_missobjects(int missobjects) { missobjects_ = missobjects; }
    void inc_missobjects(int missobjects) { missobjects_ += missobjects; }
    void set_parent(Model3D* parent) { parent_ = parent; }

    /*
     *  Add a new child to the current object
     *      obj: new child to add on the objects_ array
     *  Return a raw pointer to the new object inserted
     */
    inline Model3D* add_child(std::unique_ptr<Model3D> obj) noexcept
    {
        /* Only non moved object can have child and grid */
        if (IsMoved()) {
            return nullptr;
        }

        auto obj_raw = obj.get();

        obj->set_parent(this);

        lock();
        if (obj->type() == kMODEL3D_CAMERA) {
            objects_.insert(objects_.begin(), std::move(obj));
        } else {
            objects_.push_back(std::move(obj));
        }
        unlock();

        obj_raw->ComputePlacements();

        /* Dont decrement if Wall or Camera */
        if (obj_raw->type() != kMODEL3D_WALL &&
            obj_raw->type() != kMODEL3D_CAMERA) {
            --missobjects_;
        }

        return obj_raw;
    }

    /*
     *  Add new grid parent coords for current object
     *  Apply these add to the grid parent
     */
    inline void add_placement(int i, int j, int k) noexcept
    {
        lock();
        placements_.push_back({i,j,k});
        unlock();

        parent_->AddItemToGrid(i, j, k, this);
    }

    /*
     *  Clear placements_ array for current Object
     *  Apply theses removes to the grid parent
     */
    inline void clear_placements() noexcept
    {
        for (auto &p : placements_) {
            parent_->RemoveItemToGrid(p[0], p[1], p[2], this);
        }
        placements_.clear();
    }

    /*
     *  Delegate Mutators 
     */
    inline void InverseMove() noexcept
    {
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
     *  Pivot around the current object for detect collision with neighbors.
     */
    void PivotCollision() noexcept;

    /*
     *  Check if location_object point is inside the current Model
     */
    bool IsInside (glm::vec3 location_object) const;

    /*
     *  Return a list of childs which are found in i,j,k coords
     *  in the Grid of the current object.
     *  If i,j,k are outside of the Grid, the search occurs in othet near objects.
     */
    std::vector<Model3D*> FindItemsInGrid(int i, int j, int k) const noexcept;

    /*
     *  Return a list of neighbors qualified for a collision with current object
     */
    std::vector<Model3D*> GetCollisionNeighbors() const noexcept;

    /*
     *  Return a first side neighbor
     */
    Model3D* GetNeighborSide(int side) const noexcept;

    /*
     *  Check if target must be eligible for neighbors with clipping constraint defined by "level" parameter
     */
    bool IsClippingNear(Model3D* target, int level) const noexcept;

    /*
     *  Clear each sides neighbors_ list
     */
    void ResetNeighbors() noexcept;

    /*
     *  Add a new Child Object to the Grid
     *      i,j,k: coords into Grid array
     *      obj: child object
     */
    void AddItemToGrid(int i, int j, int k, Model3D *obj) noexcept;

    /*
     *  Remove Child Object to the Grid
     *      i,j,k: coords into Grid array
     *      obj: child object
     */
    void RemoveItemToGrid(int i, int j, int k, Model3D *obj) noexcept;

    /*
     *  Remove child and return the unique_ptr associated to this one
     */
    std::unique_ptr<Model3D> TransfertChild(Model3D* child) noexcept;

    /*
     *  Return the first point of the grid
     */
    glm::vec3 GetGrid0() const noexcept;

    /*
     *  Compute placements coords in the parent grid
     */
    void ComputePlacements() noexcept;

    /*
     *  Compute all childs placements coords
     */
    void ComputeChildsPlacements() noexcept;

    /*
     *  Flush all items into the grid
     */
    void ResetGrid() noexcept;

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
     *  neighbors of current 3d model, indexed by
     *      placement index
     *      side constant
     *      Model3D id
     */
    std::vector<std::map<int, Model3D*>> neighbors_{static_cast<int>(kSIDES)};

    /*
     *  Parent of the current 3d model
     */
    Model3D* parent_{nullptr};

    /*
     *  Grid placements for the childs of the model
     */
    std::vector<Model3D*> ***grid_{nullptr};

    /*
     *  Placements coordinates into the grid
     *  of the parent model for current model
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
