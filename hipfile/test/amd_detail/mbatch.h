/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "batch/batch.h"
#include "hipfile-warnings.h"

#include <gmock/gmock.h>

#include <stdexcept>
#include <queue>

/*
 * A place to create mocks for the batch module.
 */

namespace hipFile {

class MBatchOperation : public IBatchOperation {
public:
    MBatchOperation() = default;

    inline static std::queue<std::shared_ptr<MBatchOperation>>& get_queue() {
        HIPFILE_WARN_NO_EXIT_DTOR_OFF
        static std::queue<std::shared_ptr<MBatchOperation>> mocked_ops;
        HIPFILE_WARN_NO_EXIT_DTOR_ON
        return mocked_ops;
    }

    inline static constexpr auto MBatchOpMaker = [](std::unique_ptr<const hipFileIOParams_t> p, std::shared_ptr<IBuffer> b, std::shared_ptr<IFile> f) {
        // Discard params
        (void) p;
        (void) b;
        (void) f;
        return std::make_shared<MBatchOperation>();
    };
    // OR
    inline static constexpr auto MBatchOpMaker_queue = [](std::unique_ptr<const hipFileIOParams_t> p, std::shared_ptr<IBuffer> b, std::shared_ptr<IFile> f) {
        // Discard params
        (void) p;
        (void) b;
        (void) f;
        auto& mocked_ops = get_queue();
        if (mocked_ops.empty()) {
            throw std::runtime_error("Testing error: No mocks available to construct.");
        }
        auto op = mocked_ops.front();
        mocked_ops.pop();
        return op;
    };
};

class MBatchContext : public IBatchContext {
public:
    MOCK_METHOD(unsigned, get_capacity, (), (const, noexcept, override));
    MOCK_METHOD(void, submit_operations, (const hipFileIOParams_t *params, const unsigned num_params, const BatchOpMaker& make_op),
                (override));
    MOCK_METHOD(std::unordered_set<std::shared_ptr<IBatchOperation>>,  get_ops, (), (override));
};

}
