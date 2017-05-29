/*
* Room class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_ROOM_H_
#define ENGINE_UNIVERSE_ROOM_H_

#include <vector>
#include <memory>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <tbb/mutex.h>

#include "engine/universe/model3d.h"
#include "engine/physics/collision_engine.h"
#include "engine/universe/wall.h"
#include "engine/universe/brick.h"
#include "engine/universe/camera.h"

namespace engine {
namespace universe {

/* Define a Room (with walls and moving objects) in the 3d universe */
class Room : public Model3D {

public:
    /* Grid Constants */
    static constexpr int kGRID_Y = 6;
    static constexpr int kGRID_X = 8;
    static constexpr int kGRID_Z = 8;
    static constexpr float kGRID_UNIT = 2.0f;
    /* Room Side Constants */
    static constexpr int kFLOOR = 0;
    static constexpr int kROOF = 1;
    static constexpr int kLEFT = 2;
    static constexpr int kRIGHT = 3;
    static constexpr int kFRONT = 4;
    static constexpr int kBACK = 5;

    /* Constructors */
    Room();
    Room(glm::vec4 location);
    Room(glm::vec4 location, std::unique_ptr<Camera> cam);
    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows,
         std::unique_ptr<Camera> cam);

    /* Accessors */
    glm::vec4 location() const { return location_; }
    Camera * cam() const { return cam_; }
    int countObjects() const { return objects_.size(); }
    inline int countMovingObjects() const {
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
    /* Mutators */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

    /* New hop functions */
    void Draw(Camera *cam);
    void DetectCollision(std::vector<Room*> neighbors);
    /* Object Generation functions */
    void GenerateRandomObject();
    void GenerateWalls();
    void GenerateObjects();
    /* Grid state functions */
    std::vector<std::unique_ptr<Model3D>> ReinitGrid();
    std::unique_ptr<Model3D> TransfertObject(std::unique_ptr<Model3D> obj, bool force);
    std::vector<std::unique_ptr<Model3D>> ListOutsideObjects();
    void MoveCamera() { if (cam_ != nullptr) cam_->Move(); };

private:
    /* Room attributes */
    glm::vec4 location_;
    std::vector<Model3D*> grid_[kGRID_Y][kGRID_X][kGRID_Z];
    std::vector<std::unique_ptr<Model3D>> objects_;
    tbb::mutex room_mutex_;
    tbb::mutex grid_mutex_;
    std::vector<bool> doors_{false, false, false, false, false, false};
    std::vector<bool> windows_{false, false, false, false, false, false};
    int nbobjects_{33};
    Camera *cam_{nullptr};

    /* Some internal functions */
    Model3D *GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale);
    void PivotCollision(Model3D *object, std::vector<Room*> neighbors);
    void DisplayGrid();
    bool IsInRoom (glm::vec3 location_object) const;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_ROOM_H_
