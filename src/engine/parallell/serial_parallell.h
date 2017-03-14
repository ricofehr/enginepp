#ifndef ENGINE_PARALLELL_SERIALPARALLELL_H_
#define ENGINE_PARALLELL_SERIALPARALLELL_H_

#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace parallell {

class SerialParallell : public EngineParallell {

public:
    void InitCollisionParallell() override final;
    float ComputeCollisionParallell(float box1[], float box2[]) override final;
    ~SerialParallell() override = default;

};

}//namespace parallell
}//namespace engine

#endif //ENGINE_PARALLELL_SERIALPARALLELL_H_
