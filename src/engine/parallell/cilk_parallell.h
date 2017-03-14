#ifndef ENGINE_PARALLELL_CILKPARALLELL_H_
#define ENGINE_PARALLELL_CILKPARALLELL_H_

#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace parallell {

class CilkParallell : public EngineParallell {

public:
    void InitCollisionParallell() override final;
    float ComputeCollisionParallell(float box1[], float box2[]) override final;
    ~CilkParallell() override = default;

};

}//namespace parallell
}//namespace engine

#endif //ENGINE_PARALLELL_CILKPARALLELL_H_
