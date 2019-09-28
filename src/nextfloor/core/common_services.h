/**
 *  @file common_services.h
 *  @brief CommonServices Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_COMMONSERVICES_H_
#define NEXTFLOOR_CORE_COMMONSERVICES_H_

#include <cassert>

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/timer.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/window_settings.h"

#include "nextfloor/objects/camera.h"

#include "nextfloor/factory/facade_factory.h"

namespace nextfloor {

namespace core {

/**
 *  @class CommonServices
 *  @brief CommonServices is an unique entry point who provides global services\n
 *  to other program modules\n
 *  Implement ServiceLocator Pattern
 */
class CommonServices {

public:
    /*
     *  Service Accessors
     */
    static ConfigParser& getConfig() { return Instance()->config(); }

    static const WindowSettings& getWindowSettings() { return Instance()->window_settings(); }

    static const FileIO& getFileIO() { return Instance()->fileIO(); }

    static Timer& getTimer() { return Instance()->timer(); }

    static const Log& getLog() { return Instance()->log(); }

    static const RandomGenerator& getRandomGenerator() { return Instance()->random_generator(); }

    static const Exit& getExit() { return Instance()->exit(); }

    static nextfloor::factory::FacadeFactory& getFactory() { return Instance()->factory(); }

    static void initWindowSettings(nextfloor::renderer::SceneWindow* window) { Instance()->makeWindowSettings(window); }

protected:
    CommonServices();
    ~CommonServices() = default;

    CommonServices(CommonServices&&) = default;
    CommonServices& operator=(CommonServices&&) = default;
    CommonServices(const CommonServices&) = delete;
    CommonServices& operator=(const CommonServices&) = delete;

private:
    void Init();

    /**
     *  Return sole Instance
     *  @return sole CommonServices instance
     */
    static CommonServices* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new CommonServices;
        return instance;
    }

    ConfigParser& config()
    {
        assert(config_ != nullptr);
        return *(config_.get());
    }

    const WindowSettings& window_settings()
    {
        assert(window_settings_ != nullptr);
        return *(window_settings_.get());
    }

    const FileIO& fileIO()
    {
        assert(file_io_ != nullptr);
        return *(file_io_.get());
    }

    Timer& timer()
    {
        assert(timer_ != nullptr);
        return *(timer_.get());
    }

    const Log& log() const
    {
        assert(log_ != nullptr);
        return *(log_.get());
    }

    const RandomGenerator& random_generator() const
    {
        assert(random_generator_ != nullptr);
        return *(random_generator_.get());
    }

    const Exit& exit() const
    {
        assert(exit_ != nullptr);
        return *(exit_.get());
    }

    nextfloor::factory::FacadeFactory& factory() const
    {
        assert(factory_ != nullptr);
        return *(factory_.get());
    }

    void makeWindowSettings(nextfloor::renderer::SceneWindow* window)
    {
        window_settings_ = factory_->MakeWindowSettings(window);
    }

    std::unique_ptr<ConfigParser> config_{nullptr};
    std::unique_ptr<WindowSettings> window_settings_{nullptr};
    std::unique_ptr<FileIO> file_io_{nullptr};
    std::unique_ptr<Timer> timer_{nullptr};
    std::unique_ptr<Log> log_{nullptr};
    std::unique_ptr<RandomGenerator> random_generator_{nullptr};
    std::unique_ptr<Exit> exit_{nullptr};
    std::unique_ptr<nextfloor::factory::FacadeFactory> factory_{nullptr};
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_COMMONSERVICES_H_
