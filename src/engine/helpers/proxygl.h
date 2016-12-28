#ifndef ENGINE_HELPERS_PROXYGL_H_
#define ENGINE_HELPERS_PROXYGL_H_

namespace engine {

/* Forward declaration of class Universe */
namespace universe {
    class Universe;
}//namespace universe

namespace helpers {
/* Namespace who init GL environment */
namespace proxygl {

/* fixed height and width for gl window */
constexpr float kWidthWindow = 1200.0f;
constexpr float kHeightWindow = 740.0f;

void initGL();
void settingsGL(engine::universe::Universe *universe);

}//namespace proxygl
}//namespace helpers
}//namespace engine

#endif //ENGINE_HELPERS_PROXYGL_H_
