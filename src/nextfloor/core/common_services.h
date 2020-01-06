/**
 *  @file common_services.h
 *  @brief CommonServices Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_COMMONSERVICES_H_
#define NEXTFLOOR_CORE_COMMONSERVICES_H_

#include <memory>

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/exit.h"

#include "nextfloor/core/core_factory.h"

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
    static ConfigParser* getConfig();
    static const FileIO* getFileIO();
    static const Log* getLog();
    static const RandomGenerator* getRandomGenerator();
    static const Exit* getExit();

protected:
    CommonServices(const CoreFactory& factory);
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
    static CommonServices* Instance();

    ConfigParser* config();
    const FileIO* fileIO();
    const Log* log() const;
    const RandomGenerator* random_generator() const;
    const Exit* exit() const;

    std::unique_ptr<ConfigParser> config_{nullptr};
    std::unique_ptr<FileIO> file_io_{nullptr};
    std::unique_ptr<Log> log_{nullptr};
    std::unique_ptr<RandomGenerator> random_generator_{nullptr};
    std::unique_ptr<Exit> exit_{nullptr};
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_COMMONSERVICES_H_
