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

#include "llvm/Config/llvm-config.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"

#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR >= 11
#include "llvm/ADT/StringRef.h"
#endif

// work around ObjectLinkingLayer -> RTDyldObjectLinkingLayer rename
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 4
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#else
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#endif

#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

#include <memory>

// use legacy interfaces in LLVM 8 and 9
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR >= 8

#define RTDYLDOBJECTLINKINGLAYER LegacyRTDyldObjectLinkingLayer
#define IRCOMPILELAYER LegacyIRCompileLayer
#define LOCALCXXRUNTIMEOVERRIDES LegacyLocalCXXRuntimeOverrides

#else

#define RTDYLDOBJECTLINKINGLAYER RTDyldObjectLinkingLayer
#define IRCOMPILELAYER IRCompileLayer
#define LOCALCXXRUNTIMEOVERRIDES LocalCXXRuntimeOverrides

#endif

namespace llvm {
namespace orc {

class KaleidoscopeJIT {
public:
// work around ModuleHandleT changes in LLVM 7
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR >= 7
  ExecutionSession ES;
  std::shared_ptr<SymbolResolver> Resolver;
  typedef RTDYLDOBJECTLINKINGLAYER ObjLayerT;
  typedef IRCOMPILELAYER<ObjLayerT, SimpleCompiler> CompileLayerT;
  typedef VModuleKey ModuleHandleT;
  KaleidoscopeJIT()
      : Resolver(createLegacyLookupResolver(
            ES,
            #if LLVM_VERSION_MAJOR < 11
            [this](const std::string &Name) -> JITSymbol {
            #else
            [this](llvm::StringRef Name) -> JITSymbol {
            #endif
              #if LLVM_VERSION_MAJOR < 11
              if (auto Sym = CompileLayer.findSymbol(Name, false))
              #else
              if (auto Sym = CompileLayer.findSymbol(Name.str(), false))
              #endif
                return Sym;
              else if (auto Err = Sym.takeError())
                return std::move(Err);
              if (auto SymAddr =
              #if LLVM_VERSION_MAJOR < 11
                      RTDyldMemoryManager::getSymbolAddressInProcess(Name))
              #else
                      RTDyldMemoryManager::getSymbolAddressInProcess(Name.str()))
              #endif
                return JITSymbol(SymAddr, JITSymbolFlags::Exported);
              return nullptr;
            },
            [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
        TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        ObjectLayer(ES,
                    [this](VModuleKey) {
                      return RTDYLDOBJECTLINKINGLAYER::Resources{
                          std::make_shared<SectionMemoryManager>(), Resolver};
                    }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); })
        {
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
        }

  VModuleKey addModule(std::unique_ptr<Module> M)
    {
    // Add the module to the JIT with a new VModuleKey.
    auto K = ES.allocateVModule();
    cantFail(CompileLayer.addModule(K, std::move(M)));
    ModuleHandles.push_back(K);
    return K;
    }

  void removeModule(VModuleKey K)
    {
    cantFail(CompileLayer.removeModule(K));
    }

  JITSymbol findSymbol(const std::string Name)
    {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
    }

  JITTargetAddress getSymbolAddress(const std::string Name)
    {
    return cantFail(findSymbol(Name).getAddress());
    }

#elif defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 4
  typedef RTDyldObjectLinkingLayer ObjLayerT;
  typedef IRCompileLayer<ObjLayerT, SimpleCompiler> CompileLayerT;
  typedef CompileLayerT::ModuleHandleT ModuleHandleT;

  KaleidoscopeJIT()
      : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        ObjectLayer([]() { return std::make_shared<SectionMemoryManager>(); }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); })
        {
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
        }

    ModuleHandleT addModule(std::unique_ptr<Module> M) {
      // Build our symbol resolver:
      // Lambda 1: Look back into the JIT itself to find symbols that are part of
      //           the same "logical dylib".
      // Lambda 2: Search for external symbols in the host process.
      auto Resolver = createLambdaResolver(
          [&](const std::string &Name) {
            if (auto Sym = CompileLayer.findSymbol(Name, false))
              return Sym;
            return JITSymbol(nullptr);
          },
          [](const std::string &Name) {
            if (auto SymAddr =
                  RTDyldMemoryManager::getSymbolAddressInProcess(Name))
              return JITSymbol(SymAddr, JITSymbolFlags::Exported);
            return JITSymbol(nullptr);
          });

      // Add the set to the JIT with the resolver we created above and a newly
      // created SectionMemoryManager.
      auto H =  cantFail(CompileLayer.addModule(std::move(M),
                                                std::move(Resolver)));
      ModuleHandles.push_back(H);
      return H;
    }

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  JITTargetAddress getSymbolAddress(const std::string Name) {
    return cantFail(findSymbol(Name).getAddress());
  }

  void removeModule(ModuleHandleT H) {
    cantFail(CompileLayer.removeModule(H));
  }
#else
  typedef ObjectLinkingLayer<> ObjLayerT;
  typedef IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef CompileLayerT::ModuleSetHandleT ModuleHandleT;

  KaleidoscopeJIT()
      : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); })
      {
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
      }

  ModuleHandleT addModule(std::unique_ptr<Module> M) {
    // We need a memory manager to allocate memory and resolve symbols for this
    // new module. Create one that resolves symbols by looking back into the
    // JIT
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR == 4
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    auto Resolver = createLambdaResolver(
        [&](const std::string &Name) {
          if (auto Sym = CompileLayer.findSymbol(Name, false))
            return Sym;
          return JITSymbol(nullptr);
        },
        [](const std::string &Name) {
          if (auto SymAddr =
                RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return JITSymbol(SymAddr, JITSymbolFlags::Exported);
          return JITSymbol(nullptr);
        });
#else
    auto Resolver = createLambdaResolver(
        [&](const std::string &Name) {
          if (auto Sym = findMangledSymbol(Name))
            return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
          return RuntimeDyld::SymbolInfo(nullptr);
        },
        [](const std::string &S) { return nullptr; });
#endif
    auto H = CompileLayer.addModuleSet(singletonSet(std::move(M)),
                                       make_unique<SectionMemoryManager>(),
                                       std::move(Resolver));

    ModuleHandles.push_back(H);
    return H;
  }

  void removeModule(ModuleHandleT H) {
    ModuleHandles.erase(
        std::find(ModuleHandles.begin(), ModuleHandles.end(), H));
    CompileLayer.removeModuleSet(H);
  }

  JITSymbol findSymbol(const std::string Name) {
    return findMangledSymbol(mangle(Name));
  }
#endif


  TargetMachine &getTargetMachine() { return *TM; }

private:

  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  template <typename T> static std::vector<T> singletonSet(T t) {
    std::vector<T> Vec;
    Vec.push_back(std::move(t));
    return Vec;
  }

  JITSymbol findMangledSymbol(const std::string &Name) {
    // Search modules in reverse order: from last added to first added.
    // This is the opposite of the usual search order for dlsym, but makes more
    // sense in a REPL where we want to bind to the newest available definition.
    for (auto H : make_range(ModuleHandles.rbegin(), ModuleHandles.rend()))
      if (auto Sym = CompileLayer.findSymbolIn(H, Name, true))
        return Sym;

    // If we can't find the symbol in the JIT, try looking in the host process.
    if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
      return JITSymbol(SymAddr, JITSymbolFlags::Exported);

    return nullptr;
  }

  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  std::vector<ModuleHandleT> ModuleHandles;

  orc::LOCALCXXRUNTIMEOVERRIDES CXXRuntimeOverrides;
};

} // End namespace orc.
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

