#ifndef ENGINE_PARALLELL_ENGINEPARALLELL_H_
#define ENGINE_PARALLELL_ENGINEPARALLELL_H_

namespace engine {
namespace parallell {

class EngineParallell {

public:
    virtual void InitCollisionParallell() = 0;
    virtual float ComputeCollisionParallell(float box1[], float box2[]) = 0;
    virtual ~EngineParallell() = default;

    static const int kPARALLELL_SERIAL = 1;
    static const int kPARALLELL_CILK = 2;
    static const int kPARALLELL_CL = 3;
};

}//namespace parallell
}//namespace engine

#endif //ENGINE_PARALLELL_ENGINEPARALLELL_H_
