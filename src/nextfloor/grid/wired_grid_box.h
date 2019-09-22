/**
 *  @file wired_grid_box.h
 *  @brief WiredGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_WIREDGRIDBOX_H_
#define NEXTFLOOR_GRID_WIREDGRIDBOX_H_

#include "nextfloor/objects/grid_box.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class WiredGridBox
 *  @brief Abstract class who defines generic grid object
 */
class WiredGridBox : public nextfloor::objects::GridBox {

public:
    WiredGridBox(const glm::vec3& coords, nextfloor::objects::Grid* owner);
    ~WiredGridBox() override = default;

    WiredGridBox(WiredGridBox&&) = default;
    WiredGridBox& operator=(WiredGridBox&&) = default;
    WiredGridBox(const WiredGridBox&) = delete;
    WiredGridBox& operator=(const WiredGridBox&) = delete;

    void add(nextfloor::objects::Mesh* object) final;
    void remove(nextfloor::objects::Mesh* object) final;
    void clear() final;
    std::vector<nextfloor::objects::Mesh*> other_occupants(const nextfloor::objects::Mesh& object) const final;

    bool IsInto(const nextfloor::objects::Mesh& object) const final;
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

    std::vector<nextfloor::objects::Mesh*> occupants() const final
    {
        std::vector<nextfloor::objects::Mesh*> occupants;
        for (auto& occupant : occupants_) {
            auto meshes = occupant->AllStubMeshs();
            occupants.insert(occupants.end(), meshes.begin(), meshes.end());
        }

        return occupants;
    }

protected:
    nextfloor::objects::Mesh* getFirstOccupant()
    {
        if (occupants_.size() == 0) {
            return nullptr;
        }

        return occupants_[0];
    }

private:
    std::vector<nextfloor::objects::Mesh*> occupants_;
    nextfloor::objects::Grid* owner_;
    glm::vec3 coords_;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_GRIDBOX_H_
