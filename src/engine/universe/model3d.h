/*
* Model3D class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_MODEL3D_H_
#define ENGINE_UNIVERSE_MODEL3D_H_

#include <memory>
#include <tbb/mutex.h>

#include "engine/geometry/shape3d.h"
#include "engine/geometry/box.h"
#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace universe {

/* Forward declaration of Camera class */
class Camera;

/* Define a 3d model */
class Model3D {

public:
    Model3D();
    /* Default move and copy constructor / operator */
    Model3D(Model3D&&) = default;
    Model3D& operator=(Model3D&&) = default;

    Model3D(const Model3D&) = default;
    Model3D& operator=(const Model3D&) = default;

    void PrepareDraw(Camera *cam);
    void Draw();
    std::vector<Model3D*> DetectCollision(Model3D *obstacle, tbb::mutex &collision_mutex,
                                          engine::parallell::EngineParallell *proxy_parallell);

    int id() const { return id_; }
    float distance() const { return distance_; }
    Model3D *obstacle() { return obstacle_; }
    engine::geometry::Box border() const { return border_; }

    void set_distance(float distance) { distance_ = distance; }
    void set_obstacle(Model3D *obstacle) { obstacle_ = obstacle; }

    bool IsMoved() const { return border_.IsMoved(); }
    bool IsCrossed() const { return is_crossed_; }
    bool IsControlled() const { return is_controlled_; }
    friend bool operator==(const Model3D &o1, const Model3D &o2);
    friend bool operator!=(const Model3D &o1, const Model3D &o2);
    int type() const { return type_; }

    /* Default destructor */
    virtual ~Model3D() = default;

protected:
    static const int kMODEL3D_CAMERA = 1;
    static const int kMODEL3D_WALL = 2;
    static const int kMODEL3D_WINDOW = 3;
    static const int kMODEL3D_DOOR = 4;
    static const int kMODEL3D_BRICK = 5;

    std::vector<std::unique_ptr<engine::geometry::Shape3D>> elements_;
    engine::geometry::Box border_;
    Model3D *obstacle_;
    float distance_;
    int id_;
    int id_last_collision_;
    int type_;
    bool is_crossed_;
    bool is_controlled_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_MODEL3D_H_
