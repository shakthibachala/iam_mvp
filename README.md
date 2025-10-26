# IAM MVP Architecture & Folder Structure

## 1. Architecture Design Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                            │
├──────────────────────────────────────────────────────────────────┤
│  Orchestration        Observability         Visualization         │
│  ├─ Pipeline         ├─ Traces             ├─ Feature Maps      │
│  ├─ DAG Execute      ├─ Metrics            ├─ Attribution Viz   │
│  └─ Chain Compose    └─ Logging            └─ Circuit Display   │
│  (Kill LangChain)    (Kill LangSmith)      (Kill Gradio)        │
└────────────┬──────────────────┬─────────────────┬────────────────┘
             │                  │                  │
┌────────────▼──────────────────▼─────────────────▼────────────────┐
│                         VECTOR LAYER                              │
├──────────────────────────────────────────────────────────────────┤
│  CLT/Features         Attribution           Steering/Audit        │
│  ├─ Encode/Decode    ├─ IntGrad            ├─ Vector Construct  │
│  ├─ TopK Activate    ├─ Feature Select     ├─ Policy Apply      │
│  └─ Skip Connect     └─ Circuit Extract    └─ Safety Verify     │
│  (Kill SAELens)      (Kill Captum)         (Kill Petri)         │
└────────────┬──────────────────┬─────────────────┬────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
         C++ Core           C++ Core           C++ Core
             │                  │                  │
┌────────────▼──────────────────▼─────────────────▼────────────────┐
│                      MLIR COMPILER LAYER                          │
├──────────────────────────────────────────────────────────────────┤
│  CLT Dialect          Attribution Dialect   Policy Dialect        │
│  ├─ iam.clt.encode   ├─ iam.attr.grad     ├─ iam.policy.steer  │
│  ├─ iam.clt.decode   ├─ iam.attr.path     ├─ iam.policy.verify │
│  └─ CLTFusionPass    └─ AttrCachePass     └─ PolicyProjectPass  │
└────────────┬──────────────────┬─────────────────┬────────────────┘
             └──────────────────┼──────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  StableHLO → XLA → TPU │
                    └────────────────────────┘
```

## 2. Folder Structure

```
iam/
├── setup.py
├── CMakeLists.txt
├── pyproject.toml
│
├── python/
│   ├── iam/
│   │   ├── __init__.py
│   │   │
│   │   ├── application/
│   │   │   ├── __init__.py
│   │   │   ├── orchestration/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── dag_executor.py
│   │   │   │   └── chain_composer.py
│   │   │   ├── observability/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── tracer.py
│   │   │   │   ├── metrics.py
│   │   │   │   └── logger.py
│   │   │   └── visualization/
│   │   │       ├── __init__.py
│   │   │       ├── feature_maps.py
│   │   │       ├── attribution_viz.py
│   │   │       └── circuit_display.py
│   │   │
│   │   ├── vector/
│   │   │   ├── __init__.py
│   │   │   ├── clt/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transcoder.py
│   │   │   │   ├── activation.py
│   │   │   │   └── training.py
│   │   │   ├── attribution/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── integrated_gradients.py
│   │   │   │   ├── feature_selector.py
│   │   │   │   └── circuit_extractor.py
│   │   │   └── steering/
│   │   │       ├── __init__.py
│   │   │       ├── vector_constructor.py
│   │   │       ├── policy_applier.py
│   │   │       └── safety_verifier.py
│   │   │
│   │   └── core/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       └── types.py
│   │
│   └── bindings/
│       ├── __init__.py
│       ├── clt_bindings.cpp
│       ├── attribution_bindings.cpp
│       └── steering_bindings.cpp
│
├── cpp/
│   ├── include/
│   │   ├── iam/
│   │   │   ├── core/
│   │   │   │   ├── Tensor.hpp
│   │   │   │   ├── Memory.hpp
│   │   │   │   └── Types.hpp
│   │   │   ├── clt/
│   │   │   │   ├── Transcoder.hpp
│   │   │   │   ├── Encoder.hpp
│   │   │   │   ├── Decoder.hpp
│   │   │   │   └── TopKActivation.hpp
│   │   │   ├── attribution/
│   │   │   │   ├── IntegratedGradients.hpp
│   │   │   │   ├── FeatureSelector.hpp
│   │   │   │   ├── CircuitExtractor.hpp
│   │   │   │   └── Graph.hpp
│   │   │   └── steering/
│   │   │       ├── VectorConstructor.hpp
│   │   │       ├── PolicyApplier.hpp
│   │   │       ├── SafetyVerifier.hpp
│   │   │       └── Compositor.hpp
│   │   └── pybind/
│   │       └── Exports.hpp
│   │
│   └── src/
│       ├── clt/
│       ├── attribution/
│       └── steering/
│
├── mlir/
│   ├── include/
│   │   └── iam/
│   │       ├── Dialect/
│   │       │   ├── IAMDialect.hpp
│   │       │   ├── CLTOps.hpp
│   │       │   ├── AttributionOps.hpp
│   │       │   └── PolicyOps.hpp
│   │       └── Passes/
│   │           ├── CLTFusionPass.hpp
│   │           ├── AttributionCachePass.hpp
│   │           ├── PolicyProjectionPass.hpp
│   │           └── LowerToStableHLO.hpp
│   │
│   ├── lib/
│   │   ├── Dialect/
│   │   │   ├── IAMDialect.cpp
│   │   │   ├── CLTOps.cpp
│   │   │   ├── AttributionOps.cpp
│   │   │   └── PolicyOps.cpp
│   │   └── Passes/
│   │       ├── CLTFusionPass.cpp
│   │       ├── AttributionCachePass.cpp
│   │       └── PolicyProjectionPass.cpp
│   │
│   └── tablegen/
│       ├── IAMBase.td
│       ├── CLTOps.td
│       ├── AttributionOps.td
│       ├── PolicyOps.td
│       └── IAMTypes.td
│
├── notebooks/
│   ├── 01_pythia_clt_training.ipynb
│   ├── 02_attribution_demo.ipynb
│   ├── 03_steering_hello_world.ipynb
│   └── 04_scaling_gemmascope.ipynb
│
└── tests/
    ├── unit/
    │   ├── test_clt.py
    │   ├── test_attribution.py
    │   └── test_steering.py
    ├── integration/
    │   ├── test_mlir_lowering.py
    │   └── test_pipeline.py
    └── benchmarks/
        ├── perf_clt.py
        └── memory_profile.py
```


# IAM MVP Implementation Plan - 20 Steps

## Phase 1: Foundation (Steps 1-5)

### Step 1: Build System & Project Setup
**Build**: CMake configuration, PyBind11 setup, MLIR dependencies
**Files**: `CMakeLists.txt`, `setup.py`, `pyproject.toml`
**Test**: Verify `pip install -e .` works, MLIR links correctly
**Integration**: Foundation for all components

### Step 2: MLIR Dialect Base
**Build**: IAM dialect registration, basic types
**Files**: `mlir/tablegen/IAMBase.td`, `mlir/include/iam/Dialect/IAMDialect.hpp`
**Test**: MLIR round-trip test, dialect loads
**Integration**: Compiler foundation for all ops

### Step 3: CLT MLIR Operations
**Build**: Define iam.clt.encode/decode ops in TableGen
**Files**: `mlir/tablegen/CLTOps.td`, `mlir/lib/Dialect/CLTOps.cpp`
**Test**: Op verification, type checking
**Integration**: Enables CLT fusion optimization

### Step 4: Attribution MLIR Operations
**Build**: Define iam.attr.grad/path ops
**Files**: `mlir/tablegen/AttributionOps.td`, `mlir/lib/Dialect/AttributionOps.cpp`
**Test**: Gradient flow verification
**Integration**: Attribution graph construction

### Step 5: Policy MLIR Operations
**Build**: Define iam.policy.steer/verify ops
**Files**: `mlir/tablegen/PolicyOps.td`, `mlir/lib/Dialect/PolicyOps.cpp`
**Test**: Policy application verification
**Integration**: Steering vector application

## Phase 2: C++ Core (Steps 6-10)

### Step 6: Core Types & Memory
**Build**: Tensor wrapper, memory pool
**Files**: `cpp/include/iam/core/Tensor.hpp`, `cpp/include/iam/core/Memory.hpp`
**Test**: Memory leak detection, allocation benchmarks
**Integration**: Shared by all C++ components

### Step 7: CLT Transcoder C++
**Build**: Skip-transcoder with TopK activation
**Files**: `cpp/include/iam/clt/Transcoder.hpp`, `cpp/include/iam/clt/TopKActivation.hpp`
**Test**: Shape correctness, sparsity=32
**Integration**: Core feature extraction

### Step 8: Attribution C++ Core
**Build**: Integrated gradients implementation
**Files**: `cpp/include/iam/attribution/IntegratedGradients.hpp`, `cpp/include/iam/attribution/Graph.hpp`
**Test**: Gradient accumulation, path verification
**Integration**: Feature importance computation

### Step 9: Steering C++ Core
**Build**: Vector construction, policy application
**Files**: `cpp/include/iam/steering/VectorConstructor.hpp`, `cpp/include/iam/steering/PolicyApplier.hpp`
**Test**: Vector norm, composition test
**Integration**: Runtime steering mechanism

### Step 10: PyBind11 Exports
**Build**: Python bindings for C++ classes
**Files**: `python/bindings/clt_bindings.cpp`, `python/bindings/attribution_bindings.cpp`, `python/bindings/steering_bindings.cpp`
**Test**: Python import test, type conversion
**Integration**: Python-C++ bridge

## Phase 3: Vector Layer Python (Steps 11-13)

### Step 11: CLT Python Module
**Build**: Python wrapper for CLT transcoder
**Files**: `python/iam/vector/clt/transcoder.py`, `python/iam/vector/clt/training.py`
**Test**: Train 10 steps on dummy data
**Integration**: Replaces SAELens

### Step 12: Attribution Python Module
**Build**: Attribution API, feature selection
**Files**: `python/iam/vector/attribution/integrated_gradients.py`, `python/iam/vector/attribution/feature_selector.py`
**Test**: Feature importance ranking
**Integration**: Replaces Captum

### Step 13: Steering Python Module
**Build**: Steering vectors, safety verification
**Files**: `python/iam/vector/steering/vector_constructor.py`, `python/iam/vector/steering/safety_verifier.py`
**Test**: Contrastive activation test
**Integration**: Replaces Petri audit

## Phase 4: Application Layer (Steps 14-16)

### Step 14: Orchestration Module
**Build**: Pipeline DAG executor
**Files**: `python/iam/application/orchestration/pipeline.py`, `python/iam/application/orchestration/dag_executor.py`
**Test**: Sequential execution test
**Integration**: Replaces LangChain

### Step 15: Observability Module
**Build**: Tracer, metrics collection
**Files**: `python/iam/application/observability/tracer.py`, `python/iam/application/observability/metrics.py`
**Test**: Trace capture verification
**Integration**: Replaces LangSmith

### Step 16: Visualization Module
**Build**: Feature maps, attribution display
**Files**: `python/iam/application/visualization/feature_maps.py`, `python/iam/application/visualization/attribution_viz.py`
**Test**: Output format validation
**Integration**: Replaces Gradio

## Phase 5: Optimization & Integration (Steps 17-20)

### Step 17: CLT Fusion Pass
**Build**: MLIR optimization fusing CLT with forward pass
**Files**: `mlir/lib/Passes/CLTFusionPass.cpp`, `mlir/include/iam/Passes/CLTFusionPass.hpp`
**Test**: Before/after performance comparison
**Integration**: 5x speedup enabler

### Step 18: Pythia-170m Integration
**Build**: Model loader, activation hooks
**Files**: `notebooks/01_pythia_clt_training.ipynb`
**Test**: Model loads, activations extracted
**Integration**: Hello world demonstration

### Step 19: End-to-End Pipeline
**Build**: Complete flow CLT→Attribution→Steering
**Files**: `notebooks/03_steering_hello_world.ipynb`
**Test**: Full pipeline execution
**Integration**: MVP validation

### Step 20: Gemmascope-2b Scaling
**Build**: Scale testing, performance benchmarks
**Files**: `notebooks/04_scaling_gemmascope.ipynb`, `tests/benchmarks/perf_clt.py`
**Test**: Memory <5GB, latency <500ms
**Integration**: Proves production readiness