// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

//===----- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include <memory>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"

#pragma GCC diagnostic pop

#if !defined LLVM_VERSION_MAJOR || LLVM_VERSION_MAJOR <= 9
#error Unsupported LLVM version
#endif

namespace llvm
    {
namespace orc
    {
class KaleidoscopeJIT
    {
    public:
    std::unique_ptr<ExecutionSession> ES;
    RTDyldObjectLinkingLayer ObjectLayer;
    IRCompileLayer CompileLayer;

    DataLayout DL;
    MangleAndInterner Mangle;
    ThreadSafeContext Ctx;
    JITDylib& mainJD;
    SectionMemoryManager* memory_manager = nullptr;

    KaleidoscopeJIT(std::unique_ptr<ExecutionSession> ES,
                    JITTargetMachineBuilder JTMB,
                    DataLayout DL)
        : ES(std::move(ES)), ObjectLayer(*this->ES,
                                         [&]()
                                         {
                                             auto smgr = std::make_unique<SectionMemoryManager>();
                                             memory_manager = smgr.get();
                                             return smgr;
                                         }),
          CompileLayer(
              *this->ES,
              ObjectLayer,
              std::make_unique<ConcurrentIRCompiler>(ConcurrentIRCompiler(std::move(JTMB)))),
          DL(std::move(DL)), Mangle(*this->ES, this->DL), Ctx(std::make_unique<LLVMContext>()),
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 10
          mainJD(this->ES->createBareJITDylib("<main>"))
#else
            mainJD(this->ES->createJITDylib("<main>"))
#endif
        {
        mainJD.addGenerator(
            cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix())));
        }

    ~KaleidoscopeJIT()
        {
        // avoid seg fault when an exception is thrown later
        // https://github.com/taichi-dev/taichi/issues/655#issuecomment-620344230
        // https://github.com/taichi-dev/taichi/pull/885/files#
        if (memory_manager)
            memory_manager->deregisterEHFrames();
        }

    const DataLayout& getDataLayout() const
        {
        return DL;
        }

    static std::unique_ptr<KaleidoscopeJIT> Create()
        {
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 12
        auto EPC = SelfExecutorProcessControl::Create();
        if (!EPC)
            return nullptr;

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));
#else
        auto ES = std::make_unique<ExecutionSession>();
#endif

        auto JTMB = JITTargetMachineBuilder::detectHost();

        if (!JTMB)
            return nullptr;

        auto DL = JTMB->getDefaultDataLayoutForTarget();

        if (!DL)
            return nullptr;

        return std::make_unique<KaleidoscopeJIT>(std::move(ES), std::move(*JTMB), std::move(*DL));
        }

    Error addModule(std::unique_ptr<Module> M)
        {
        return CompileLayer.add(mainJD, ThreadSafeModule(std::move(M), Ctx));
        }

    Expected<JITEvaluatedSymbol> findSymbol(std::string Name)
        {
        return ES->lookup({&mainJD}, Mangle(Name));
        }

    JITTargetAddress getSymbolAddress(const std::string Name)
        {
        return findSymbol(Name)->getAddress();
        }
    };

    } // End namespace orc.
    } // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
