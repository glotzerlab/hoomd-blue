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

#include <utility>
#include <memory>

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
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

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
        ExecutionSession ES;
        RTDyldObjectLinkingLayer ObjectLayer;
        IRCompileLayer CompileLayer;

        DataLayout DL;
        MangleAndInterner Mangle;
        ThreadSafeContext Ctx;
        JITDylib &mainJD;

    KaleidoscopeJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
        : ObjectLayer(ES,
                        []() { return std::make_unique<SectionMemoryManager>(); }),
            CompileLayer(ES, ObjectLayer, std::make_unique<ConcurrentIRCompiler>(ConcurrentIRCompiler(std::move(JTMB)))),
            DL(std::move(DL)), Mangle(ES, this->DL),
            Ctx(std::make_unique<LLVMContext>()),
            #if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 10
            mainJD(this->ES.createBareJITDylib("<main>"))
            #else
            mainJD(this->ES.createJITDylib("<main>"))
            #endif
        {
        mainJD.addGenerator(
            cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix())));
    }
    const DataLayout &getDataLayout() const { return DL; }

    static std::unique_ptr<KaleidoscopeJIT> Create() {
    auto JTMB = JITTargetMachineBuilder::detectHost();

    // if (!JTMB)
    //     throw std::runtime_error("Error initializing JITTargetMachineBuilder");

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    // if (!DL)
    //     throw std::runtime_error("Error initializing DataLayout");

    return std::make_unique<KaleidoscopeJIT>(std::move(*JTMB), std::move(*DL));
    }

  Error addModule(std::unique_ptr<Module> M) {
    return CompileLayer.add(mainJD, ThreadSafeModule(std::move(M), Ctx));
  }

  Expected<JITEvaluatedSymbol> findSymbol(std::string Name) {
    return ES.lookup({&mainJD}, Mangle(Name));
  }

    JITTargetAddress getSymbolAddress(const std::string Name)
        {
        return findSymbol(Name)->getAddress();
        }
    };

    } // End namespace orc.
    } // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
