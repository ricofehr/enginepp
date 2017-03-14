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

void InitGL();
void SettingsGL(engine::universe::Universe *universe);

}//namespace proxygl
}//namespace helpers
}//namespace engine

#endif //ENGINE_HELPERS_PROXYGL_H_
