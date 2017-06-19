/**
 *  @file border.h
 *  @brief Border class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef ENGINE_GRAPHICS_BORDER_H_
#define ENGINE_GRAPHICS_BORDER_H_

#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "engine/graphics/cube.h"

/**
 *  @namespace engine
 *  @brief Common parent namespace
 */
namespace engine {

/**
 *  @namespace engine::graphics
 *  @brief Elementary 3d graphic classes
 */
namespace graphics {

/**
 *  @class Border
 *  @brief Each 3d object in the scene must be fill into a border.\n
 *  This border is represented by a Cube object with a Delegator scheme.\n
 *  Used for 3d objects coordinates and collision compute.
 */
class Border {

public:

    /**
     *  Constructor
     */
    Border();

    /**
     *  Constructor
     *  @param scale is a float factor applied to the native coords array
     *  @param location is the center point of the Border
     */
    Border(float scale, glm::vec4 location);

    /**
     *  Constructor
     *  @param scale is a 3 axis factor applied to the native coords array
     *  @param location is the center point of the Border
     */
    Border(glm::vec3 scale, glm::vec4 location);

    /**
     *  Constructor
     *  @param scale is a float factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param move is the translation vector
     */
    Border(float scale, glm::vec4 location, glm::vec4 move);

    /**
     *  Constructor
     *  @param scale is a 3 axis factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param move is the translation vector
     */
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move);

    /**
     *  Constructor
     *  @param scale is a float factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param coords is the native coords of the cube border
     */
    Border(float scale, glm::vec4 location, std::vector<glm::vec3> coords);

    /**
     *  Constructor
     *  @param scale is a 3 axis factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param coords is the native coords of the cube border
     */
    Border(glm::vec3 scale, glm::vec4 location, std::vector<glm::vec3> coords);

    /**
     *  Constructor
     *  @param scale is a float factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param move is the translation vector
     *  @param coords is the native coords of the cube border
     */
    Border(float scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    /**
     *  Constructor
     *  @param scale is a 3 axis factor applied to the native coords array
     *  @param location is the center point of the Border
     *  @param move is the translation vector
     *  @param coords is the native coords of the cube border
     */
    Border(glm::vec3 scale, glm::vec4 location, glm::vec4 move, std::vector<glm::vec3> coords);

    /**
     *  Default Move constructor
     */
    Border(Border&&) = default;

    /**
     *  Default Move assignment
     */
    Border& operator=(Border&&) = default;

    /**
     *  Copy constructor Deleted
     *  Because cube_ is unique_ptr
     */
    Border(const Border&) = delete;

    /**
     *  Copy assignment Deleted
     *  Because cube_ is unique_ptr
     */
    Border& operator=(const Border&) = delete;

    /**
     *  Default destructor
     */
    ~Border() = default;

    /**
     *  Compute and return coords computed with current model matrix
     *  @return border coords computed
     */
    std::vector<glm::vec3> ComputeCoords() const;

    /**
     *  Compute new location for the next move
     */
    void MoveCoords();

    /**
     *  Delegate Move
     */
    void MoveLocation() { cube_->MoveLocation(); }

    /*
     *  Delegate Accessors
     */
    bool IsMoved() const { return cube_->IsMoved(); }
    int IsMovedX() const { return cube_->IsMovedX(); }
    int IsMovedY() const { return cube_->IsMovedY(); }
    int IsMovedZ() const { return cube_->IsMovedZ(); }
    glm::vec4 location() const { return cube_->location(); }
    glm::vec3 scale() const { return cube_->scale(); }
    glm::vec4 move() const { return cube_->move(); }
    float distance() const { return cube_->distance(); }

    /*
     *  Delegate Mutators
     */
    void set_distance(float distance) { cube_->set_distance(distance); }
    void set_move(glm::vec3 move) { cube_->set_move(move); }
    void InverseMove() { cube_->InverseMove(); }

private:

    /** Represents the border */
    std::unique_ptr<Cube> cube_{nullptr};

    /** Border coords */
    std::vector<glm::vec3> coords_;
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_GRAPHICS_BORDER_H_

