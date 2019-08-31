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

#include "nextfloor/factory/mesh_factory.h"
#include "nextfloor/factory/hid_factory.h"
#include "nextfloor/factory/command_factory.h"

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
    static ConfigParser* getConfig() { return Instance()->config(); }

    static const FileIO* getFileIO() { return Instance()->fileIO(); }

    static Timer* getTimer() { return Instance()->timer(); }

    static const Log* getLog() { return Instance()->log(); }

    static const RandomGenerator* getRandomGenerator() { return Instance()->random_generator(); }

    static const Exit* getExit() { return Instance()->exit(); }

    static nextfloor::factory::MeshFactory* getMeshFactory() { return Instance()->mesh_factory(); }

    static const nextfloor::factory::HidFactory* getHidFactory() { return Instance()->hid_factory(); }

    static const nextfloor::factory::CommandFactory* getCommandFactory() { return Instance()->command_factory(); }

protected:
    CommonServices();

    CommonServices(CommonServices&&) = default;
    CommonServices& operator=(CommonServices&&) = default;
    CommonServices(const CommonServices&) = delete;
    CommonServices& operator=(const CommonServices&) = delete;

    ~CommonServices() = default;

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

    ConfigParser* config()
    {
        assert(config_ != nullptr);
        return config_.get();
    }

    const FileIO* fileIO()
    {
        assert(file_io_ != nullptr);
        return file_io_.get();
    }

    Timer* timer()
    {
        assert(timer_ != nullptr);
        return timer_.get();
    }

    const Log* log()
    {
        assert(log_ != nullptr);
        return log_.get();
    }

    const RandomGenerator* random_generator()
    {
        assert(random_generator_ != nullptr);
        return random_generator_.get();
    }

    const Exit* exit()
    {
        assert(exit_ != nullptr);
        return exit_.get();
    }

    nextfloor::factory::MeshFactory* mesh_factory()
    {
        assert(mesh_factory_ != nullptr);
        return mesh_factory_.get();
    }

    const nextfloor::factory::HidFactory* hid_factory()
    {
        assert(hid_factory_ != nullptr);
        return hid_factory_.get();
    }

    const nextfloor::factory::CommandFactory* command_factory()
    {
        assert(command_factory_ != nullptr);
        return command_factory_.get();
    }

    std::unique_ptr<ConfigParser> config_{nullptr};
    std::unique_ptr<FileIO> file_io_{nullptr};
    std::unique_ptr<Timer> timer_{nullptr};
    std::unique_ptr<Log> log_{nullptr};
    std::unique_ptr<RandomGenerator> random_generator_{nullptr};
    std::unique_ptr<Exit> exit_{nullptr};
    std::unique_ptr<nextfloor::factory::MeshFactory> mesh_factory_{nullptr};
    std::unique_ptr<nextfloor::factory::HidFactory> hid_factory_{nullptr};
    std::unique_ptr<nextfloor::factory::CommandFactory> command_factory_{nullptr};
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_COMMONSERVICES_H_
