/**
 *  @file model3d.h
 *  @brief Model3D class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_MODEL3D_H_
#define NEXTFLOOR_UNIVERSE_MODEL3D_H_

#include <memory>
#include <vector>
#include <map>
#include <tbb/mutex.h>
#include <tbb/tbb.h>
#include <glm/glm.hpp>

#include "nextfloor/graphics/shape3d.h"
#include "nextfloor/graphics/border.h"
#include "nextfloor/physics/collision_engine.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::universe
 *  @brief World elements
 */
namespace universe {

/**
 *  @class Model3D
 *  @brief Abstract class who defines generic 3d object and shall be inherited by all other 3d models
 */
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
     *  Initial collision countdown value
     */
    static constexpr int kCOLLISION_COUNTDOWN = 4;

    /**
     *  Default Move constructor
     */
    Model3D(Model3D&&) = default;

    /**
     *  Default Move assignment
     */
    Model3D& operator=(Model3D&&) = default;

    /**
     *  Copy constructor Deleted
     *  Because border_ is unique_ptr
     */
    Model3D(const Model3D&) = delete;

    /**
     *  Copy assignment Deleted
     *  Because border_ is unique_ptr
     */
    Model3D& operator=(const Model3D&) = delete;

    /**
     *  Destructor
     *  Deallocates grid_ raw pointers array
     */
    virtual ~Model3D();

    /*
     *  (In)Equality Operators
     */
    friend bool operator==(const Model3D& o1, const Model3D& o2);
    friend bool operator!=(const Model3D& o1, const Model3D& o2);

    /**
     *  Compute neighbors in respect with the clipping constraint
     *  All objects too far or hidden by current view are not included.
     *  @param level is clipping level (1 -> high clipping, 2 -> low clipping)
     *  @return vector of neighbors
     */
    std::vector<Model3D*> FindClippingNeighbors(int level) const noexcept;

    /**
     *  Record HID events
     */
    inline virtual void RecordHID() noexcept {
        if (get_camera() != nullptr) {
            objects_[0]->RecordHID();
        }
    }

    /**
     *  Detect collision for current object and for his childs
     */
    void DetectCollision() noexcept;

    /**
     *  Proceed to move current object
     */
    virtual void Move() noexcept;

    /**
     *  Draw the object and his childs
     */
    inline virtual void Draw() noexcept
    {
        /* Draw current object */
        for (auto &element : elements_) {
            element->Draw();
        }

        /* Draw childs objects */
        for (auto &object : objects_) {
            object->Draw();
        }
    }

    /**
     *  Test if a position in the grid of current object is empty
     *  @param i,j,k are the coords of the position in the grid
     *  @return kGRID_UNKNOW if coords are not valid\n
     *          kGRID_EMPTY if empty\n
     *          kGRID_USED if there is objects in this position
     */
    int IsPositionInTheGridEmpty(int i, int j, int k) const noexcept;

    /**
     *  Display the Grid onto stdout for Debug or Information purpose
     */
    void DisplayGrid() const noexcept;

    /**
     *  Test if 2 objects are in same direction
     *  @param target is the other object to compare
     *  @return true if the both are moving in same direction
     */
    inline bool IsSameDirectionThan(const Model3D* target)
    {
        if (IsMovedX() != target->IsMovedX()) {
            return false;
        }

        if (IsMovedY() != target->IsMovedY()) {
            return false;
        }

        if (IsMovedZ() != target->IsMovedZ()) {
            return false;
        }

        return true;
    }

    /*
     *  Accessors
     */
    int id() const { return id_; }
    Model3D* obstacle() const { return obstacle_; }
    Model3D* parent() { return parent_; }
    nextfloor::graphics::Border* border() const { return border_.get(); }
    std::vector<std::vector<int>> placements() const { return placements_; }
    constexpr int gridx() const { return grid_x_; }
    constexpr int gridy() const { return grid_y_; }
    constexpr int gridz() const { return grid_z_; }
    constexpr float grid_unitx() const { return grid_unit_x_; }
    constexpr float grid_unity() const { return grid_unit_y_; }
    constexpr float grid_unitz() const { return grid_unit_z_; }
    inline bool IsFull() const { return missobjects_ <= 0 || objects_.size() >= grid_x_ * grid_y_ * grid_z_; }
    virtual int countChilds() const { return objects_.size(); }
    glm::vec3 direction() const { return direction_; }
    glm::vec3 head() const { return head_; }

    /**
     *  Return Speed (scale factor of move)
     */
    float get_speed() { return 3.5f / nextfloor::graphics::Shape3D::sMoveFactor; }

    /**
     *  Compute the first point of the grid
     *  @return coords of the first point
     */
    inline glm::vec3 GetGrid0() const noexcept
    {
        return location() - glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2);
    }

    /**
     *  Return Camera object if present into childs array
     *  @return raw pointer to the camera if present, or nullptr
     */
    inline Model3D* get_camera() const noexcept
    {
        if (countChilds() > 0) {
            if (objects_[0]->IsCamera()) {
                return objects_[0].get();
            } else {
                for (auto &object: objects_) {
                    auto camera = object->get_camera();
                    if (camera != nullptr) {
                        return camera;
                    }
                }
            }
        }

        return nullptr;
    }

    /**
     *  Return number of Moving Object into childs
     *  @return count of moving childs
     */
    inline virtual int countMovingChilds() const
    {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, objects_.size()),
            0,
            [&](const tbb::blocked_range<int>& r, int init)->int {
                for (int a = r.begin(); a != r.end(); ++a) {
                    if (objects_[a]->IsMoved()) {
                        ++init;
                    }
                }
                return init;
            },
            [](int x, int y)->int {
                return x + y;
            }
        );
    }

    /*
     *  Delegate Accessors
     */
    glm::vec3 location() const { return glm::vec3(border_->location()); }
    bool IsMoved() const { return border_->IsMoved(); }
    int IsMovedX() const { return border_->IsMovedX(); }
    int IsMovedY() const { return border_->IsMovedY(); }
    int IsMovedZ() const { return border_->IsMovedZ(); }
    float distance() const { return border_->distance(); }

    /*
     *  Mutators
     */
    void set_obstacle(Model3D* obstacle) { obstacle_ = obstacle; }
    void reset_missobjects(int missobjects) { missobjects_ = 0; }
    void set_missobjects(int missobjects) { missobjects_ = missobjects; }
    void inc_missobjects(int missobjects) { missobjects_ += missobjects; }
    void set_parent(Model3D* parent) { parent_ = parent; }
    void flag_collision_with_camera() { is_collision_with_camera_ = true; }
    void set_move(glm::vec3 move_vector) { border_->set_move(move_vector); }

    /**
     *  Add a new child to the current object
     *  @param obj is the new child to add
     *  @return a raw pointer to the new object inserted
     */
    inline Model3D* add_child(std::unique_ptr<Model3D> obj) noexcept
    {
        /* Only non moved object can have child and grid */
        if (IsMoved()) {
            return nullptr;
        }

        auto obj_raw = obj.get();

        obj_raw->set_parent(this);

        lock();
        auto initial_objects_size = objects_.size();
        /* Insert Camera as first element. Push on the last for others */
        if (obj_raw->IsCamera()) {
            objects_.insert(objects_.begin(), std::move(obj));
        } else {
            objects_.push_back(std::move(obj));
        }
        /* Ensure object is inserted */
        assert(objects_.size() == initial_objects_size + 1);
        unlock();

        obj_raw->ComputePlacements();

        /* Dont decrement if Wall or Camera */
        if (!obj_raw->IsWall() && !obj_raw->IsCamera()) {
            --missobjects_;
        }

        return obj_raw;
    }

    /**
     *  Set collision distance on the border and shape elements
     *  @param distance before the collision between this object and other
     */
    inline void set_distance(float distance) {
        border_->set_distance(distance);
        for (auto &element : elements_) {
            element->set_distance(distance);
        }
    }

    /**
     *  Inverse the Move direction of the current object
     */
    inline void InverseMove() noexcept
    {
        border_->InverseMove();
        for (auto &e : elements_) {
            e->InverseMove();
        }
    }

    /**
     *  Lock a mutex on the current object
     */
    void lock() { object_mutex_.lock(); }

    /**
     *  Unlock a mutex on the current object
     */
    void unlock() { object_mutex_.unlock(); }

protected:

    /**
     *  Constructor
     *  Protected scope ensures Abstract Class Design
     */
    Model3D();

    /**
     *  Allocate grid array dynamically
     *  Use of raw pointers
     */
    void InitGrid();

    /**
     *  Return a list of all neighbors of current object
     *  @return a vector of neighbors
     */
    std::vector<Model3D*> FindAllNeighbors() const noexcept;

    /**
     *  Return a list of childs which are found with coords
     *  in the Grid of the current object.
     *  If coords are outside of the Grid, the search occurs in other near objects.
     *  @param i,j,k are the coords into the current object grid
     *  @return a vector of 3d objects
     */
    std::vector<Model3D*> FindItemsInGrid(int i, int j, int k) const noexcept;

    /**
     *  Pivot around the current object for detect collision with neighbors.
     */
    void PivotCollision() noexcept;

    /**
     *  Return a list of neighbors qualified for a collision with current object
     *  @return a vector of neighbors
     */
    std::vector<Model3D*> FindCollisionNeighbors() const noexcept;

    /**
     *  Return all neighbors of the current object following a side constraint
     *  @param side is the side constraint
     *  @return a vector of neighbors
     */
    std::vector<Model3D*> FindNeighborsSide(int side) const noexcept;

    /**
     *  Compute placements coords of the current object in the parent grid
     */
    void ComputePlacements() noexcept;

    /**
     *  Add a new coords child into the Grid
     *  @param i,j,k are the coords into the current object grid
     *  @param obj is the child object
     */
    void AddItemToGrid(int i, int j, int k, Model3D* obj) noexcept;

    /**
     *  Remove a child placement to the Grid
     *  @param i,j,k are the coords into the current object grid
     *  @param obj is the child object
     */
    void RemoveItemToGrid(int i, int j, int k, Model3D* obj) noexcept;

    /**
     *  Check if location_object point is inside the current object
     *  @param location_object is the point to test
     *  @return true if the point is inside the current object
     */
    bool IsInside (glm::vec3 location_object) const;

    /**
     *  Flush all items into the grid
     */
    void ResetGrid() noexcept;

    /*
     *  Accessors
     */
    int id_last_collision() const { return id_last_collision_; }
    bool IsUniverse() const { return type_ == kMODEL3D_UNIVERSE; }
    bool IsRoom() const { return type_ == kMODEL3D_ROOM; }
    bool IsCamera() const { return type_ == kMODEL3D_CAMERA; }
    bool IsWall() const { return type_ == kMODEL3D_WALL; }
    bool IsCameraCollision() const { return type_ == kMODEL3D_CAMERA || is_collision_with_camera_; }

    /**
     *  Add new grid parent coords for current object
     *  Apply these add to the grid parent
     *  @param i,j,k are the placement coords into parent grid
     */
    inline void add_placement(int i, int j, int k) noexcept
    {
        lock();
        placements_.push_back({i,j,k});
        unlock();

        parent_->AddItemToGrid(i, j, k, this);
    }

    /**
     *  Clear placements array for current object
     *  Apply theses removes to the grid parent
     */
    inline void clear_placements() noexcept
    {
        for (auto &p : placements_) {
            parent_->RemoveItemToGrid(p[0], p[1], p[2], this);
        }

        lock();
        placements_.clear();
        unlock();
    }


    /** 3d shapes which composes the current object */
    std::vector<std::unique_ptr<nextfloor::graphics::Shape3D>> elements_;

    /** the box which defines the border */
    std::unique_ptr<nextfloor::graphics::Border> border_{nullptr};

    /** childs of the current object */
    std::vector<std::unique_ptr<Model3D>> objects_;

    /** size in gl pixels for one unity of X grid axis */
    float grid_unit_x_{0.0f};

    /** size in gl pixels for one unity of Y grid axis */
    float grid_unit_y_{0.0f};

    /** size in gl pixels for one unity of Z grid axis */
    float grid_unit_z_{0.0f};

    /** number of X unities in the grid (grid_x * grid_unit_x = grid width) */
    int grid_x_{0};

    /** number of Y unities in the grid (grid_y * grid_unit_y = grid height) */
    int grid_y_{0};

    /** number of Z unities in the grid (grid_z * grid_unit_z = grid depth) */
    int grid_z_{0};

    /** Type of 3d object */
    int type_{10000};

    /** VIew direction vector */
    glm::vec3 direction_;

    /** Head coords */
    glm::vec3 head_;

private:

    /**
     *  Select the Collision Engine Algorithm
     */
    void InitCollisionEngine();

    /**
     *  Check if target must be eligible for neighbors with clipping constraint
     *  @param target is the neighbor to test
     *  @param level is the clipping level
     *  @return true if eligible
     */
    bool IsClippingNear(Model3D* target, int level) const noexcept;

    /**
     *  Compute an array of sides qualified with a moving direction
     *  @param dirx,diry,dirz are axis directions of the moving current object
     *  @return a vector of sides constant
     */
    const std::vector<int> ListSidesInTheDirection(int dirx, int diry, int dirz) const noexcept;

    /**
     *  Compute the right grid coords following a side
     *  This function dont use class members.
     *  @param i,j,k are initial coords into Grid array
     *  @param side is the side constant targetting
     *  @return the new coords after apply side constraint
     */
    glm::vec3 GetNeighborCoordsBySide(int i, int j, int k, int side) const;

    /**
     *  Return the side place in the current grid object following coords
     *  @param i,j,k are the coords into the current object grid
     *  @return the border side constant
     */
    int BeInTheRightPlace(int i, int j, int k) const;

    /**
     *  Return only one (first finded) neighbor of the current object following a side constraint
     *  @param side is the side constraint
     *  @return a neighbor
     */
    Model3D* FindNeighborSide(int side) const noexcept;

    /**
     *  Remove child off the current object and return the unique_ptr associated to this one
     *  @param child to remove
     *  @return the unique_ptr to the old child
     */
    std::unique_ptr<Model3D> TransfertChild(Model3D* child) noexcept;


    /** Parent of the current 3d model */
    Model3D* parent_{nullptr};

    /** Grid placements for the childs of the current object */
    std::vector<Model3D*> ***grid_{nullptr};

    /** Placements coordinates into the parent grid */
    std::vector<std::vector<int>> placements_;

    /** If setted, obstacle_ is the collision partner */
    Model3D* obstacle_{nullptr};

    /** Mutex ensures thread safe instructions */
    tbb::mutex object_mutex_;

    /** Engine used for collision computes */
    nextfloor::physics::CollisionEngine* collision_engine_{nullptr};

    /** Unique id */
    int id_;

    /** Last obstacle id */
    int id_last_collision_;

    /** Frames count before test again collision with id_last_collision */
    int collision_countdown_{0};

    /** Count of childs who are still missing */
    int missobjects_{0};

    /** Collision with Camera is a special case and need to be flagged */
    bool is_collision_with_camera_{false};
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_MODEL3D_H_
