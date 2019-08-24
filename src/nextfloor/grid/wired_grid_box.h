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
    WiredGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner);

    WiredGridBox(WiredGridBox&&) = default;
    WiredGridBox& operator=(WiredGridBox&&) = default;
    WiredGridBox(const WiredGridBox&) = default;
    WiredGridBox& operator=(const WiredGridBox&) = default;

    virtual ~WiredGridBox() override = default;

    virtual void add(nextfloor::objects::Mesh* object) noexcept override;
    virtual void remove(nextfloor::objects::Mesh* object) noexcept override;
    virtual void clear() noexcept override;
    virtual std::vector<nextfloor::objects::Mesh*> other_occupants(nextfloor::objects::Mesh* object) noexcept override
    {
        std::vector<nextfloor::objects::Mesh*> others(0);
        for (auto& occupant : occupants_) {
            if (occupant->id() != object->id()) {
                others.push_back(occupant);
            }
        }

        return others;
    }

    virtual bool IsInside(nextfloor::objects::Mesh* object) const noexcept override;
    virtual bool IsEmpty() const noexcept override;
    virtual bool IsFilled() const noexcept override;

    virtual bool IsFrontPositionFilled() const noexcept override;
    virtual bool IsRightPositionFilled() const noexcept override;
    virtual bool IsBackPositionFilled() const noexcept override;
    virtual bool IsLeftPositionFilled() const noexcept override;
    virtual bool IsBottomPositionFilled() const noexcept override;
    virtual bool IsTopPositionFilled() const noexcept override;

    virtual int size() const noexcept override { return occupants_.size(); }
    virtual glm::vec3 coords() const noexcept override { return coords_; }
    virtual std::vector<nextfloor::objects::Mesh*> occupants() noexcept override { return occupants_; }


    // virtual std::vector<nextfloor::objects::Mesh*> FindCollisionNeighbors(nextfloor::objects::Mesh* target)
    // noexcept override
    // {
    //     auto neighbors = owner_->FindCollisionNeighbors();
    //     sort(neighbors.begin(), neighbors.end());
    //     neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto new_neighbors = FindFrontPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontRightPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontRightBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontRightTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontLeftPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontLeftBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindFrontLeftTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindRightPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindRightBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindRightTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackRightPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackRightBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackRightTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackLeftPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackLeftBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBackLeftTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindLeftPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindLeftBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindLeftTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindBottomPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     new_neighbors = FindTopPositionCollisionNeighbors(target);
    //     neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
    //     sort(neighbors.begin(), neighbors.end());
    //     neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    // }

protected:
    nextfloor::objects::Mesh* getFirstOccupant()
    {
        if (occupants_.size() == 0) {
            return nullptr;
        }

        return occupants_[0];
    }

private:
    // std::vector<nextfloor::objects::Mesh*> FindFrontPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontRightPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontRightPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontRightBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontRightBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontRightTopPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontRightTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontLeftPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontLeftPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontLeftBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontLeftBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontLeftTopPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontLeftTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindFrontBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindFrontTopPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindFrontTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindRightPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindRightPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindRightBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindRightBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindRightTopPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindRightTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindBackPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackRightPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackRightPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackRightBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackRightBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackRightTopPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackRightTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindBackLeftPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackLeftPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackLeftBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackLeftBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackLeftTopPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackLeftTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindBackBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindBackTopPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBackTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindLeftPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindLeftPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*>
    // FindLeftBottomPositionCollisionNeighbors(nextfloor::objects::Mesh* target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindLeftBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindLeftTopPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindLeftTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindBottomPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindBottomPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    // std::vector<nextfloor::objects::Mesh*> FindTopPositionCollisionNeighbors(nextfloor::objects::Mesh*
    // target) const
    // {
    //     std::vector<nextfloor::objects::Mesh*> neighbors(0);
    //     auto front_neighbors = owner_->FindTopPositionCollisionNeighbors(coords_);

    //     for (auto &neighbor : front_neighbors) {
    //         if (neighbor->id() == target->id()) {
    //             continue;
    //         }
    //         auto vector_neighbor = neighbor->location() - target->location();
    //         if (glm::length(target->movement()) + glm::length(neighbor->movement()) >
    //         glm::length(vector_neighbor) - neighbor->diagonal() / 2.0f &&
    //             glm::dot(target->movement(), vector_neighbor) > 0) {
    //             neighbors.push_back(neighbor);
    //         }
    //     }

    //     return neighbors;
    // }

    std::vector<nextfloor::objects::Mesh*> occupants_;
    nextfloor::objects::Grid* owner_;
    glm::vec3 coords_;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_GRIDBOX_H_
