/**
 *  @file wired_grid_box.h
 *  @brief WiredGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_WIREDGRIDBOX_H_
#define NEXTFLOOR_GRID_WIREDGRIDBOX_H_

#include "nextfloor/mesh/grid_box.h"

#include <glm/glm.hpp>
#include <tbb/mutex.h>

#include "nextfloor/playground/grid.h"
#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace layout {

/**
 *  @class WiredGridBox
 *  @brief Abstract class who defines generic grid object
 */
class WiredGridBox : public nextfloor::mesh::GridBox {

public:
    WiredGridBox(const glm::vec3& coords);
    ~WiredGridBox() override = default;

    WiredGridBox(WiredGridBox&&) = delete;
    WiredGridBox& operator=(WiredGridBox&&) = delete;
    WiredGridBox(const WiredGridBox&) = delete;
    WiredGridBox& operator=(const WiredGridBox&) = delete;

    void set_owner(nextfloor::playground::Grid* owner) final;
    void add(nextfloor::mesh::Mesh* object) final;
    void remove(nextfloor::mesh::Mesh* object) final;
    void clear() final;
    std::vector<nextfloor::mesh::Mesh*> other_occupants(const nextfloor::mesh::Mesh& object) const final;

    bool IsInto(const nextfloor::mesh::Mesh& object) const final;
    bool IsEmpty() const final;
    bool IsFilled() const final;

    bool IsFrontPositionFilled() const final;
    bool IsRightPositionFilled() const final;
    bool IsBackPositionFilled() const final;
    bool IsLeftPositionFilled() const final;
    bool IsBottomPositionFilled() const final;
    bool IsTopPositionFilled() const final;

    int size() const final { return occupants_.size(); }
    glm::vec3 coords() const final { return coords_; }

    std::vector<nextfloor::mesh::Mesh*> occupants() const final;

protected:
    nextfloor::mesh::Mesh* getFirstOccupant();

private:
    std::vector<nextfloor::mesh::Mesh*> occupants_;
    nextfloor::playground::Grid* owner_{nullptr};
    glm::vec3 coords_;
    tbb::mutex mutex_;
};

}  // namespace layout

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_GRIDBOX_H_
