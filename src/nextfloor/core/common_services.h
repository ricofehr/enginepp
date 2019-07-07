/**
 *  @file common_services.h
 *  @brief CommonServices Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_COMMONSERVICES_H_
#define NEXTFLOOR_CORE_COMMONSERVICES_H_

#include <cassert>

#include "nextfloor/core/config.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/timer.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/exit.h"

#include "nextfloor/factory/engine_factory.h"

namespace nextfloor {

namespace core {

/**
 *  @class CommonServices
 *  @brief CommonServices is an unique entry point who provides global services to other program modules\n
 *  Implement ServiceLocator Pattern
 */
class CommonServices {

public:

    /*
     *  Service Accessors
     */
    static Config* getConfig()
    {
        assert(config_ != nullptr);
        return config_;
    }

    static const FileIO* getFileIO()
    {
        assert(file_io_ != nullptr);
        return file_io_;
    }

    static Timer* getTimer()
    {
        assert(timer_ != nullptr);
        return timer_;
    }

    static const Log* getLog()
    {
        assert(log_ != nullptr);
        return log_;
    }

    static const RandomGenerator* getRandomGenerator()
    {
        assert(random_generator_ != nullptr);
        return random_generator_;
    }

    static const Exit* getExit()
    {
        assert(exit_ != nullptr);
        return exit_;
    }

    static const nextfloor::factory::EngineFactory* getFactory()
    {
        assert(factory_ != nullptr);
        return factory_;
    }

    /*
     *  Service Mutators
     */
    static void provideConfig(Config *config)
    {
        config_ = config;
    }

    static void provideFileIO(FileIO *file_io)
    {
        file_io_ = file_io;
    }

    static void provideTimer(Timer *timer)
    {
        timer_ = timer;
    }

    static void provideLog(Log *log)
    {
        log_ = log;
    }

    static void provideRandomGenerator(RandomGenerator *generator)
    {
        random_generator_ = generator;
    }

    static void provideExit(Exit *exit)
    {
        exit_ = exit;
    }

    static void provideFactory(nextfloor::factory::EngineFactory *factory)
    {
        factory_ = factory;
    }

private:

    static Config* config_;
    static FileIO* file_io_;
    static Timer* timer_;
    static Log* log_;
    static RandomGenerator* random_generator_;
    static Exit* exit_;
    static nextfloor::factory::EngineFactory* factory_;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_COMMONSERVICES_H_
