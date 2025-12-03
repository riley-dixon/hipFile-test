/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}

namespace hipFile {

/*
 * Intentionally empty interface
 */
template <typename Interface>
class IFactory {
};

/*
    Factory was created with the help of an LLM
    - typename... Args - "parameter pack" of zero or more types
    - Args... - Expand the "parameter pack"
    - typename = std::enable_if_t<...> - dummy template param used as a gate
                                         used to check for function resolution
                                         Allows for trying all overloads.
                                         If no matching signature found at all: compiler error.
    - forward<Args>(args)... - "perfectly forward" all passed args into T's matched ctor.
                               aka. expand the arg type pack and arg value pack into
                               separate std::forward calls.
*/
template <
    typename Interface,
    typename Implementation
>
class GenericFactory {
public:
    // Return an instance of Implementation
    // Usage: Factory<Implementation>::make(...)
    template <
        typename... Args,
        typename = std::enable_if_t<std::is_constructible<Implementation, Args...>::value>
    >
    static Implementation make(Args&&... args)
    {
        return Implementation(std::forward<Args>(args)...);
    }

    // Return a std::shared_ptr<Interface> to an Implementation instance
    // Usage:: Factory<Interface>::make_shared(...)
    template <
        typename... Args,
        typename = std::enable_if_t<std::is_constructible<Implementation, Args...>::value>
    >
    static std::shared_ptr<Interface> make_shared(Args&&... args)
    {
        return std::static_pointer_cast<Interface>(std::make_shared<Implementation>(std::forward<Args>(args)...));
    }

    // // Return a std::shared_ptr<Interface> to an arbitrary
    // // Usage: Factory<T>::make_shared_as<BaseT>(...)
    // template <
    //     typename BaseT,
    //     typename... Args,
    //     typename = std::enable_if_t<
    //         std::is_constructible<T, Args...>::value &&
    //         std::is_base_of<BaseT, T>::value
    //     >
    // >
    // static std::shared_ptr<BaseT> make_shared_as(Args&&... args)
    // {
    //     return std::static_pointer_cast<BaseT>(std::make_shared<T>(std::forward<Args>(args)...));
    //     //return std::shared_ptr<BaseT>{new T{std::forward<Args(args)...}}
    //     //return std::make_shared<T>(std::forward<Args>(args)...);
    // }

    // // Return a std::unique_ptr of T
    // template <
    //     typename... Args,
    //     typename = std::enable_if_t<std::is_constructible<T, Args...>::value>
    // >
    // static std::unique_ptr<T> make_unique(Args&&... args)
    // {
    //     return std::make_unique<T>(std::forward<Args>(args)...);
    // }
};

struct InvalidBatchHandle : public std::invalid_argument {
    InvalidBatchHandle() : std::invalid_argument{"Invalid batch handle"}
    {
    }
};

class IBatchOperation {
public:
    virtual ~IBatchOperation() = default;
};

/// @brief Represents a single IO Request
class BatchOperation : public IBatchOperation {
public:
    /// @brief Create an operation to handle and track an IO request.
    /// @param [in] params IO parameters
    /// @param [in] buffer Buffer corresponding to params->u.batch.devPtr_base
    /// @param [in] file File corresponding params->fh
    BatchOperation(std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                   std::shared_ptr<IFile> file);

private:
    /// @brief A copy of the params provided by the application.
    /// @internal Keep this listed at the top of BatchOperation.
    const std::unique_ptr<const hipFileIOParams_t> io_params;

    /// @brief A reference to the specified Buffer.
    const std::shared_ptr<const IBuffer> buffer;

    /// @brief A reference to the specified registered File.
    const std::shared_ptr<const IFile> file;
};

template <
    typename Implementation,
    typename Interface = IBatchOperation
>
class BatchOperationFactory : public IFactory<Interface>, public GenericFactory<Interface, Implementation>
{
};

class IBatchContext {
public:
    static constexpr unsigned MAX_SIZE = 128;

    virtual ~IBatchContext()                                                                 = default;
    virtual unsigned get_capacity() const noexcept                                           = 0;
    virtual void     submit_operations(const hipFileIOParams_t *params, unsigned num_params, IFactory<IBatchOperation> factory = BatchOperationFactory<BatchOperation>{}) = 0;
};

class BatchContext : public IBatchContext {
public:
    ///
    /// @brief Return the max number of concurrent operations supported by this BatchContext.
    ///
    /// @return The max number of concurrent operations that can be processed by this BatchContext.
    /// @note This may not exceed the value returned by `MAX_SIZE`.
    unsigned get_capacity() const noexcept override;

    ///
    /// @brief Submit one or more operations to this Context.
    /// @param [in] params     Pointer to the operations to enqueue.
    /// @param [in] num_params Number of operations to enqueue.
    ///
    /// @note This is an All or None operation. If one submitted operation is not valid, no operations
    ///       will be submitted.
    ///
    void submit_operations(const hipFileIOParams_t *params, const unsigned num_params, IFactory<IBatchOperation> factory) override;

private:
    const unsigned capacity;

    /// Per-Context mutex to limit access to one caller at a time.
    /// Shared as internally we can be more strategic about concurrent access.
    mutable std::shared_mutex context_mutex;

    /// An outstanding operation is a BatchOperation that has been submitted
    /// but is not yet complete or completed but not yet retrieved by the
    /// application.
    /// shared_ptr as it may need to be passed to a backend.
    std::unordered_set<std::shared_ptr<IBatchOperation>> outstanding_ops;

    BatchContext(unsigned capacity);

    friend class BatchContextMap;
};

class BatchContextMap {
public:
    /*!
     * @brief Create a new batch context
     * @param capacity Maximum number of outstanding operations that this context can manage
     * @return An opaque handle used to reference this new batch context
     */
    hipFileBatchHandle_t createContext(unsigned capacity);

    /*!
     * @brief Destroy a batch context and release all associated resources
     * @param handle The handle for the batch context to destroy
     */
    void destroyContext(hipFileBatchHandle_t handle);

    /*!
     * @brief Get a batch context
     * @param handle The opaque handle associated with a batch context
     * @return A batch context
     */
    std::shared_ptr<IBatchContext> get(hipFileBatchHandle_t handle);

    /*!
     * @brief Clear the contents
     */
    void clear();

private:
    /// batch context lookup table
    std::unordered_map<hipFileBatchHandle_t, std::shared_ptr<IBatchContext>> active_contexts;

    /// Mutex to protect the active context map
    mutable std::shared_mutex batch_mutex;
};

static_assert(!std::is_abstract<BatchOperationFactory<BatchOperation>>::value, "Not concrete");

}
