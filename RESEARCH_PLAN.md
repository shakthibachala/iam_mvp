# IAM MVP: Chain of Thought Research & Engineering Plan

**Project**: Interpretable AI with MLIR (IAM) - Minimal Viable Product  
**Date Started**: 2025-10-26  
**Last Updated**: 2025-10-26  
**Status**: Phase 0 - Planning & Requirements Gathering

---

## Executive Summary

### Project Vision
Build a compiler-integrated interpretability framework using MLIR that replaces multiple heavyweight Python tools (SAELens, Captum, Petri, LangChain, LangSmith, Gradio) with a unified, performant, and formally verifiable system.

### Key Innovation
Treat interpretability as a **compiler problem**, not just a Python library problem. Custom MLIR dialects enable:
- Formal semantics for interpretability operations
- Compiler optimizations for 5-10x performance gains
- Provable correctness guarantees
- Hardware-agnostic execution (CPU, GPU, TPU)

### Success Criteria
- [ ] Replace SAELens for CLT/transcoder operations
- [ ] Replace Captum for attribution methods
- [ ] Replace Petri for steering/policy enforcement
- [ ] Achieve <5GB memory for Gemmascope-2B
- [ ] Achieve <500ms latency for inference + analysis
- [ ] Demonstrate on Pythia-170M as "Hello World"

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Hypothesis Tree](#2-hypothesis-tree)
3. [Research Questions & Confidence Levels](#3-research-questions--confidence-levels)
4. [Systematic Requirements Analysis](#4-systematic-requirements-analysis)
5. [Implementation Phases](#5-implementation-phases)
6. [Testing Strategy (TDD Approach)](#6-testing-strategy-tdd-approach)
7. [Progress Tracking](#7-progress-tracking)
8. [Risk Analysis & Mitigation](#8-risk-analysis--mitigation)
9. [Decision Log](#9-decision-log)
10. [Learning Resources](#10-learning-resources)

---

## 1. Current State Assessment

### ✅ What We Have

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| MLIR PoC | ✅ Working | `/iam_toy/tutorials/hw-mlir` | Basic dialect registration working |
| Build System | ✅ Working | CMakeLists.txt + build.sh | Links MLIR correctly |
| Project Structure | 📋 Designed | README.md | 20-step plan exists |
| Repository | ✅ Initialized | `.git` | Ready for development |

### ❌ What We Need

| Component | Priority | Dependencies | Risk Level |
|-----------|----------|--------------|------------|
| IAM Dialect Definition | P0 | MLIR knowledge | Medium |
| C++ Core Libraries | P0 | Eigen/xtensor | Low |
| PyBind11 Integration | P1 | C++ core | Low |
| Python API Layer | P1 | Bindings | Low |
| Test Infrastructure | P0 | None | Low |

### 🔬 Key Unknowns

1. **MLIR Dialect Design**: How to structure CLT/Attribution/Policy ops?
2. **Performance Targets**: Can we really achieve 5x speedup vs SAELens?
3. **Memory Management**: How to handle large activation tensors efficiently?
4. **StableHLO Integration**: Lowering path from IAM dialect to TPU?
5. **Python Ergonomics**: Can we match SAELens API simplicity?

---

## 2. Hypothesis Tree

### Core Hypothesis
**H0**: MLIR-based interpretability can achieve production-grade performance while maintaining research flexibility.
```
H0: MLIR enables efficient interpretability
├── H1: Custom dialects reduce overhead
│   ├── H1.1: CLT fusion eliminates Python loops [Confidence: 70%]
│   ├── H1.2: Attribution caching saves redundant compute [Confidence: 80%]
│   └── H1.3: Policy projection is zero-copy [Confidence: 60%]
│
├── H2: C++ core provides performance
│   ├── H2.1: Sparse TopK is 10x faster than Python [Confidence: 90%]
│   ├── H2.2: Memory pooling reduces allocations [Confidence: 85%]
│   └── H2.3: SIMD vectorization helps attribution [Confidence: 75%]
│
├── H3: Progressive lowering enables optimization
│   ├── H3.1: High-level IAM ops optimize independently [Confidence: 65%]
│   ├── H3.2: StableHLO path works for TPU [Confidence: 50%]
│   └── H3.3: LLVM backend handles edge cases [Confidence: 80%]
│
└── H4: API ergonomics match Python tools
    ├── H4.1: PyBind11 gives seamless integration [Confidence: 95%]
    ├── H4.2: Familiar API reduces adoption friction [Confidence: 70%]
    └── H4.3: Documentation/examples are sufficient [Confidence: 60%]
```

### Hypothesis Testing Plan

#### H1.1: CLT Fusion Eliminates Python Loops
```python
# Test Design
def test_clt_fusion_performance():
    """
    Compare:
    - SAELens: Python loop over batches
    - IAM: Single fused MLIR pass
    
    Expected: 3-5x speedup
    Measurement: Wall clock time, peak memory
    """
    baseline = benchmark_saelens_clt()
    ours = benchmark_iam_clt()
    assert ours.time < baseline.time * 0.3  # 70% faster
```

**Confidence Evolution**:
- Initial: 70% (based on MLIR literature)
- After unit tests: 75% (+5% for basic fusion working)
- After integration: 80% (+5% for real model integration)
- After benchmarks: 90% (+10% for achieving targets)

---

## 3. Research Questions & Confidence Levels

### Phase 1: Foundation (Steps 1-5)

| Question | Confidence | Evidence Needed | Current Status |
|----------|-----------|-----------------|----------------|
| Can we integrate MLIR into Python build? | 85% | CMake + pip install working | 🔬 Testing |
| Is TableGen sufficient for IAM ops? | 70% | Define 3+ ops successfully | ❌ Not started |
| Do we need custom MLIR types? | 60% | Activation tensor modeling | ❌ Not started |
| Will OpBuilder API be ergonomic enough? | 75% | Build 10+ test programs | ❌ Not started |

### Phase 2: C++ Core (Steps 6-10)

| Question | Confidence | Evidence Needed | Current Status |
|----------|-----------|-----------------|----------------|
| Which tensor library? (Eigen vs xtensor) | 50% | Performance benchmarks | ❌ Not started |
| Can we avoid copying activations? | 65% | Memory profiling | ❌ Not started |
| Is TopK on GPU worth it? | 40% | GPU vs CPU benchmark | ❌ Not started |
| How to handle variable batch sizes? | 70% | Dynamic shape tests | ❌ Not started |

### Phase 3: Vector Layer (Steps 11-13)

| Question | Confidence | Evidence Needed | Current Status |
|----------|-----------|-----------------|----------------|
| Can we match SAELens training speed? | 60% | End-to-end training benchmark | ❌ Not started |
| Is integrated gradients accurate enough? | 80% | Compare vs Captum on test cases | ❌ Not started |
| Do steering vectors compose well? | 55% | Multi-vector application tests | ❌ Not started |

### Phase 4: Application Layer (Steps 14-16)

| Question | Confidence | Evidence Needed | Current Status |
|----------|-----------|-----------------|----------------|
| Is DAG execution simpler than LangChain? | 75% | API complexity comparison | ❌ Not started |
| Can we trace without LangSmith overhead? | 80% | Tracer performance tests | ❌ Not started |
| Are matplotlib outputs sufficient? | 90% | User feedback | ❌ Not started |

### Phase 5: Optimization (Steps 17-20)

| Question | Confidence | Evidence Needed | Current Status |
|----------|-----------|-----------------|----------------|
| Does CLT fusion give 5x speedup? | 50% | Before/after benchmarks | ❌ Not started |
| Can we scale to Gemmascope-2B? | 45% | Memory profiling at scale | ❌ Not started |
| Is end-to-end pipeline reliable? | 60% | Integration tests | ❌ Not started |

---

## 4. Systematic Requirements Analysis

### 4.1 Functional Requirements

#### FR1: CLT/Transcoder Operations
```
As a researcher,
I want to extract sparse features from activations,
So that I can interpret model behavior.

Acceptance Criteria:
✅ Encode activations → sparse features (k=32)
✅ Decode features → reconstructed activations
✅ TopK selection with configurable k
✅ Skip connections for residual paths
✅ Training loop for transcoder weights

Dependencies: None (foundational)
Priority: P0 (critical path)
```

#### FR2: Attribution Methods
```
As a researcher,
I want to compute feature attributions,
So that I can identify causal relationships.

Acceptance Criteria:
✅ Integrated gradients with configurable steps
✅ Feature importance ranking
✅ Circuit extraction from attribution graph
✅ Support for multiple attribution methods

Dependencies: FR1 (needs features)
Priority: P0 (critical path)
```

#### FR3: Steering/Policy Enforcement
```
As a safety engineer,
I want to apply steering vectors,
So that I can control model behavior.

Acceptance Criteria:
✅ Construct steering vectors from contrast pairs
✅ Apply vectors at specified layers
✅ Verify policy constraints
✅ Compose multiple steering directions

Dependencies: FR1, FR2 (needs features + attribution)
Priority: P1 (important but not blocking)
```

#### FR4: Orchestration/Pipeline
```
As a practitioner,
I want to chain interpretability operations,
So that I can build complex analysis workflows.

Acceptance Criteria:
✅ DAG-based execution model
✅ Sequential and parallel composition
✅ Error handling and recovery
✅ Progress tracking and cancellation

Dependencies: FR1, FR2, FR3 (needs all vector ops)
Priority: P1 (usability)
```

### 4.2 Non-Functional Requirements

#### NFR1: Performance
```
Target: 5-10x faster than pure Python alternatives

Metrics:
- CLT encode/decode: <50ms for 1024 activations
- Integrated gradients: <200ms for attribution graph
- End-to-end pipeline: <500ms total latency

Measurement Plan:
1. Baseline benchmarks with SAELens/Captum
2. Unit-level microbenchmarks (each operation)
3. Integration benchmarks (full pipeline)
4. Continuous performance regression tests
```

#### NFR2: Memory Efficiency
```
Target: <5GB for Gemmascope-2B (65M parameters)

Strategies:
- Memory pooling for temporary tensors
- Lazy evaluation where possible
- Streaming for large activations
- Aggressive garbage collection

Measurement Plan:
1. Memory profiling with valgrind/heaptrack
2. Peak memory tracking in tests
3. Memory leak detection
4. Stress tests with large models
```

#### NFR3: Correctness
```
Target: Numerical stability and reproducibility

Strategies:
- Unit tests with known-good outputs
- Gradient checking for backprop
- Cross-validation with reference implementations
- Fuzzing for edge cases

Measurement Plan:
1. Test coverage >90%
2. Numerical accuracy tests (relative error <1e-5)
3. Integration tests against Pythia-170M
4. Continuous integration on every commit
```

#### NFR4: Usability
```
Target: API as simple as SAELens/Captum

Principles:
- Pythonic interfaces
- Sensible defaults
- Clear error messages
- Comprehensive documentation

Measurement Plan:
1. API review with potential users
2. Tutorial notebooks
3. Code examples for common tasks
4. User feedback surveys
```

---

## 5. Implementation Phases

### Phase 0: Planning & Requirements ⏳ (Current)
**Duration**: 1-2 days  
**Goal**: Complete requirements analysis and research plan

#### Deliverables
- [x] Read and understand README.md
- [x] Analyze existing MLIR PoC
- [x] Create this research plan document
- [ ] Search for MLIR best practices
- [ ] Document requirement specifications
- [ ] Create initial hypothesis tree
- [ ] Set up testing infrastructure plan

#### Confidence Tracking
- Overall Phase Confidence: **75%**
  - MLIR basics understood: 90%
  - Requirements clear: 80%
  - Testing strategy defined: 60%
  - Risk mitigation planned: 70%

---

### Phase 1: Foundation (Steps 1-5)
**Duration**: 3-5 days  
**Goal**: Build system, MLIR dialect foundation

#### Step 1: Build System & Project Setup
**Confidence: 85%** | **Priority: P0** | **Risk: Low**
```bash
# Test-Driven Development Plan
tests/test_build.py:
  ✅ test_pip_install_editable()
  ✅ test_import_iam()
  ✅ test_mlir_library_links()
  ✅ test_pybind11_module_loads()

# Implementation Plan
1. Create pyproject.toml with dependencies
2. Configure CMakeLists.txt for MLIR + PyBind11
3. Set up project structure matching README
4. Verify installation with pytest

# Success Criteria
- `pip install -e .` completes without errors
- `import iam` works in Python
- All build tests pass
```

**Files to Create**:
- `pyproject.toml`
- `setup.py`
- `CMakeLists.txt`
- `python/iam/__init__.py`
- `tests/test_build.py`

**Dependencies**: Working MLIR installation (✅ have it)

**Confidence Factors**:
- ✅ Have working MLIR PoC (+20%)
- ✅ PyBind11 is well-documented (+15%)
- ⚠️  CMake can be tricky (-10%)

---

#### Step 2: MLIR Dialect Base
**Confidence: 70%** | **Priority: P0** | **Risk: Medium**
```cpp
// Test-Driven Development Plan
mlir/test/Dialect/IAM/test_base.mlir:
  ✅ test_dialect_loads()
  ✅ test_basic_types_parse()
  ✅ test_roundtrip_module()

// Implementation Plan
1. Define IAMDialect class
2. Register with MLIR context
3. Define basic types (if needed)
4. Write MLIR round-trip test

// Success Criteria
- Dialect registers without errors
- Can parse/print basic IAM IR
- Round-trip test passes
```

**Files to Create**:
- `mlir/include/iam/Dialect/IAMDialect.hpp`
- `mlir/lib/Dialect/IAMDialect.cpp`
- `mlir/tablegen/IAMBase.td`
- `mlir/test/Dialect/IAM/test_base.mlir`

**Confidence Factors**:
- ✅ Have hello_mlir example (+15%)
- ⚠️  Need to learn TableGen (-15%)
- ⚠️  Custom types might be complex (-10%)

**Learning Required**:
- TableGen syntax and semantics
- MLIR dialect registration patterns
- Type system extensibility

---

#### Step 3: CLT MLIR Operations
**Confidence: 65%** | **Priority: P0** | **Risk: Medium**
```tablegen
// Test-Driven Development Plan
// 1. Write tests FIRST
mlir/test/Dialect/IAM/test_clt_ops.mlir:
  ✅ test_encode_op_parses()
  ✅ test_decode_op_parses()
  ✅ test_topk_op_verifies()
  ✅ test_type_checking_fails_correctly()

// 2. Then implement to make tests pass
mlir/tablegen/CLTOps.td:
  def IAM_EncodeOp : IAM_Op<"clt.encode"> {
    let summary = "Encode activations to sparse features";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$features);
  }

// Success Criteria
- All 4 CLT operations defined
- Type checking works
- Round-trip tests pass
```

**Files to Create**:
- `mlir/tablegen/CLTOps.td`
- `mlir/lib/Dialect/CLTOps.cpp`
- `mlir/include/iam/Dialect/CLTOps.hpp`
- `mlir/test/Dialect/IAM/test_clt_ops.mlir`

**Operations to Define**:
1. `iam.clt.encode` - Activation → Features
2. `iam.clt.decode` - Features → Activation
3. `iam.clt.topk` - Sparse TopK selection
4. `iam.clt.residual` - Skip connection

**Confidence Factors**:
- ✅ Clear mathematical definition (+10%)
- ⚠️  TopK operation might be tricky (-10%)
- ⚠️  Skip connections need careful design (-5%)

---

#### Step 4: Attribution MLIR Operations
**Confidence: 60%** | **Priority: P0** | **Risk: Medium**
```tablegen
// Test-Driven Development Plan
mlir/test/Dialect/IAM/test_attr_ops.mlir:
  ✅ test_grad_op_parses()
  ✅ test_path_op_verifies()
  ✅ test_circuit_op_builds_graph()

// Implementation Plan
def IAM_GradOp : IAM_Op<"attr.grad"> {
  let summary = "Compute integrated gradients";
  let arguments = (ins 
    AnyTensor:$output,
    AnyTensor:$input,
    I32Attr:$steps
  );
  let results = (outs AnyTensor:$attribution);
}
```

**Files to Create**:
- `mlir/tablegen/AttributionOps.td`
- `mlir/lib/Dialect/AttributionOps.cpp`
- `mlir/test/Dialect/IAM/test_attr_ops.mlir`

**Operations to Define**:
1. `iam.attr.grad` - Integrated gradients
2. `iam.attr.path` - Path-based attribution
3. `iam.attr.select` - Feature selection
4. `iam.attr.circuit` - Circuit extraction

**Confidence Factors**:
- ✅ Well-studied algorithm (+10%)
- ⚠️  Graph representation unclear (-15%)
- ⚠️  Caching strategy undefined (-10%)

---

#### Step 5: Policy MLIR Operations
**Confidence: 55%** | **Priority: P1** | **Risk: Medium-High**
```tablegen
// Test-Driven Development Plan
mlir/test/Dialect/IAM/test_policy_ops.mlir:
  ✅ test_steer_op_applies_vector()
  ✅ test_verify_op_checks_bounds()
  ✅ test_compose_op_combines_vectors()

// Implementation Plan
def IAM_SteerOp : IAM_Op<"policy.steer"> {
  let summary = "Apply steering vector";
  let arguments = (ins 
    AnyTensor:$activations,
    AnyTensor:$steering_vector,
    I32Attr:$layer
  );
  let results = (outs AnyTensor:$steered_activations);
}
```

**Files to Create**:
- `mlir/tablegen/PolicyOps.td`
- `mlir/lib/Dialect/PolicyOps.cpp`
- `mlir/test/Dialect/IAM/test_policy_ops.mlir`

**Operations to Define**:
1. `iam.policy.steer` - Apply steering vector
2. `iam.policy.verify` - Check safety bounds
3. `iam.policy.compose` - Combine multiple vectors
4. `iam.policy.project` - Project to valid space

**Confidence Factors**:
- ⚠️  Composition semantics unclear (-20%)
- ⚠️  Verification method undefined (-15%)
- ✅ Linear algebra is straightforward (+10%)

**Open Questions**:
- How to compose conflicting steering directions?
- What safety bounds should we verify?
- How to handle out-of-distribution vectors?

---

### Phase 2: C++ Core (Steps 6-10)
**Duration**: 5-7 days  
**Goal**: Implement core C++ functionality

#### Step 6: Core Types & Memory
**Confidence: 80%** | **Priority: P0** | **Risk: Low**
```cpp
// Test-Driven Development Plan
tests/unit/test_tensor.cpp:
  ✅ test_tensor_construction()
  ✅ test_memory_pool_reuse()
  ✅ test_zero_copy_views()
  ✅ test_no_memory_leaks()

// Implementation Plan
class Tensor {
  // Wrapper around Eigen/xtensor
  // Reference counting for views
  // Memory pool integration
};

class MemoryPool {
  // Pre-allocated memory blocks
  // RAII guarantees
  // Thread-safe allocation
};
```

**Key Decisions to Make**:
- **Tensor Library**: Eigen vs xtensor vs roll-our-own?
  - Eigen: More mature, good SIMD support
  - xtensor: NumPy-like API, better Python integration
  - Custom: Maximum control, but high development cost
  - **Decision**: Start with Eigen, evaluate xtensor later

**Files to Create**:
- `cpp/include/iam/core/Tensor.hpp`
- `cpp/include/iam/core/Memory.hpp`
- `cpp/src/core/Tensor.cpp`
- `cpp/src/core/Memory.cpp`
- `tests/unit/test_tensor.cpp`

---

#### Step 7: CLT Transcoder C++
**Confidence: 75%** | **Priority: P0** | **Risk: Medium**
```cpp
// Test-Driven Development Plan
tests/unit/test_clt.cpp:
  ✅ test_encode_shape_correct()
  ✅ test_topk_sparsity_k32()
  ✅ test_decode_reconstruction_error()
  ✅ test_skip_connection_residual()
  ✅ benchmark_encode_latency()

// Implementation Plan
class Transcoder {
public:
  Tensor encode(const Tensor& activations);
  Tensor decode(const Tensor& features);
  Tensor topk(const Tensor& logits, int k);
  
private:
  Tensor encoder_weights_;
  Tensor decoder_weights_;
  Tensor bias_;
};
```

**Performance Targets**:
- Encode: <50ms for batch_size=32, d_model=768
- Decode: <30ms (typically faster than encode)
- TopK: <10ms with k=32

**Files to Create**:
- `cpp/include/iam/clt/Transcoder.hpp`
- `cpp/include/iam/clt/TopKActivation.hpp`
- `cpp/src/clt/Transcoder.cpp`
- `cpp/src/clt/TopKActivation.cpp`
- `tests/unit/test_clt.cpp`

---

#### Step 8: Attribution C++ Core
**Confidence: 70%** | **Priority: P0** | **Risk: Medium**
```cpp
// Test-Driven Development Plan
tests/unit/test_attribution.cpp:
  ✅ test_integrated_gradients_sum_to_delta()
  ✅ test_attribution_graph_builds()
  ✅ test_path_attribution_traceback()
  ✅ test_gradient_accumulation_numerics()

// Implementation Plan
class IntegratedGradients {
public:
  Tensor compute(
    const Tensor& output,
    const Tensor& input,
    int steps = 50
  );
  
private:
  Tensor interpolate(const Tensor& baseline, 
                     const Tensor& input, 
                     float alpha);
  Tensor gradient(const Tensor& x);
};
```

**Numerical Considerations**:
- Use float32 for stability
- Accumulate in float64 to reduce errors
- Normalize attributions to sum to delta
- Handle near-zero gradients gracefully

**Files to Create**:
- `cpp/include/iam/attribution/IntegratedGradients.hpp`
- `cpp/include/iam/attribution/Graph.hpp`
- `cpp/src/attribution/IntegratedGradients.cpp`
- `cpp/src/attribution/Graph.cpp`
- `tests/unit/test_attribution.cpp`

---

#### Step 9: Steering C++ Core
**Confidence: 65%** | **Priority: P1** | **Risk: Medium**
```cpp
// Test-Driven Development Plan
tests/unit/test_steering.cpp:
  ✅ test_vector_construction_from_contrast()
  ✅ test_policy_application_modifies_activations()
  ✅ test_vector_composition_commutative()
  ✅ test_safety_bounds_verified()

// Implementation Plan
class VectorConstructor {
public:
  Tensor construct_from_contrast(
    const Tensor& positive_activations,
    const Tensor& negative_activations
  );
};

class PolicyApplier {
public:
  Tensor apply(
    const Tensor& activations,
    const Tensor& steering_vector,
    float strength = 1.0
  );
};
```

**Design Questions**:
- Should steering vectors be unit vectors?
- How to handle vector composition?
- What safety bounds to check?

**Files to Create**:
- `cpp/include/iam/steering/VectorConstructor.hpp`
- `cpp/include/iam/steering/PolicyApplier.hpp`
- `cpp/src/steering/VectorConstructor.cpp`
- `cpp/src/steering/PolicyApplier.cpp`
- `tests/unit/test_steering.cpp`

---

#### Step 10: PyBind11 Exports
**Confidence: 90%** | **Priority: P0** | **Risk: Low**
```cpp
// Test-Driven Development Plan
tests/unit/test_bindings.py:
  ✅ test_tensor_import_from_numpy()
  ✅ test_transcoder_python_api()
  ✅ test_attribution_python_api()
  ✅ test_steering_python_api()

// Implementation Plan
PYBIND11_MODULE(_iam_core, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init<py::array_t<float>>())
    .def("to_numpy", &Tensor::to_numpy);
    
  py::class_<Transcoder>(m, "Transcoder")
    .def(py::init<>())
    .def("encode", &Transcoder::encode)
    .def("decode", &Transcoder::decode);
}
```

**PyBind11 Best Practices**:
- Zero-copy NumPy integration
- Automatic GIL management
- Exception translation C++ → Python
- Documentation strings

**Files to Create**:
- `python/bindings/clt_bindings.cpp`
- `python/bindings/attribution_bindings.cpp`
- `python/bindings/steering_bindings.cpp`
- `tests/unit/test_bindings.py`

---

### Phase 3: Vector Layer Python (Steps 11-13)
**Duration**: 4-6 days  
**Goal**: Python API matching SAELens/Captum

#### Step 11: CLT Python Module
**Confidence: 80%** | **Priority: P0** | **Risk: Low**
```python
# Test-Driven Development Plan
tests/test_clt_api.py:
  ✅ test_transcoder_trains_on_dummy_data()
  ✅ test_transcoder_saves_and_loads()
  ✅ test_transcoder_inference_shape()
  ✅ test_transcoder_matches_saelens_api()

# Implementation Plan
class Transcoder:
    """Generalized sparse auto-encoder."""
    
    def __init__(self, d_model: int, d_sae: int, k: int = 32):
        self.encoder = Linear(d_model, d_sae)
        self.decoder = Linear(d_sae, d_model)
        self.k = k
    
    def encode(self, activations: Tensor) -> Tensor:
        logits = self.encoder(activations)
        return topk(logits, k=self.k)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encode(x)
        reconstructed = self.decode(features)
        return features, reconstructed
```

**API Design Principles**:
- Match SAELens API where possible
- Use type hints extensively
- Provide sensible defaults
- Clear error messages

**Files to Create**:
- `python/iam/vector/clt/transcoder.py`
- `python/iam/vector/clt/activation.py`
- `python/iam/vector/clt/training.py`
- `tests/test_clt_api.py`

---

#### Step 12: Attribution Python Module
**Confidence: 75%** | **Priority: P0** | **Risk: Low**
```python
# Test-Driven Development Plan
tests/test_attribution_api.py:
  ✅ test_integrated_gradients_basic()
  ✅ test_feature_importance_ranking()
  ✅ test_circuit_extraction()
  ✅ test_matches_captum_api()

# Implementation Plan
class IntegratedGradients:
    """Compute integrated gradients for attribution."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def attribute(
        self,
        inputs: Tensor,
        target: int,
        steps: int = 50,
        baseline: Optional[Tensor] = None
    ) -> Tensor:
        # Uses C++ backend for performance
        return _iam_core.integrated_gradients(...)
```

**Files to Create**:
- `python/iam/vector/attribution/integrated_gradients.py`
- `python/iam/vector/attribution/feature_selector.py`
- `python/iam/vector/attribution/circuit_extractor.py`
- `tests/test_attribution_api.py`

---

#### Step 13: Steering Python Module
**Confidence: 70%** | **Priority: P1** | **Risk: Medium**
```python
# Test-Driven Development Plan
tests/test_steering_api.py:
  ✅ test_steering_vector_construction()
  ✅ test_policy_application()
  ✅ test_safety_verification()
  ✅ test_vector_composition()

# Implementation Plan
class SteeringVector:
    """Construct and apply steering vectors."""
    
    @classmethod
    def from_contrast(
        cls,
        model: nn.Module,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer: int
    ) -> "SteeringVector":
        # Extract activations and compute difference
        ...
    
    def apply(
        self,
        activations: Tensor,
        strength: float = 1.0
    ) -> Tensor:
        return activations + strength * self.vector
```

**Files to Create**:
- `python/iam/vector/steering/vector_constructor.py`
- `python/iam/vector/steering/policy_applier.py`
- `python/iam/vector/steering/safety_verifier.py`
- `tests/test_steering_api.py`

---

### Phase 4: Application Layer (Steps 14-16)
**Duration**: 3-4 days  
**Goal**: High-level orchestration and tooling

#### Step 14: Orchestration Module
**Confidence: 75%** | **Priority: P1** | **Risk: Low**
```python
# Test-Driven Development Plan
tests/test_orchestration.py:
  ✅ test_pipeline_sequential_execution()
  ✅ test_dag_parallel_execution()
  ✅ test_error_handling_recovery()
  ✅ test_progress_tracking()

# Implementation Plan
class Pipeline:
    """Chain interpretability operations."""
    
    def __init__(self):
        self.steps = []
    
    def add(self, name: str, fn: Callable) -> "Pipeline":
        self.steps.append((name, fn))
        return self  # Fluent API
    
    def run(self, inputs: dict) -> dict:
        outputs = inputs.copy()
        for name, fn in self.steps:
            outputs[name] = fn(**outputs)
        return outputs
```

**Files to Create**:
- `python/iam/application/orchestration/pipeline.py`
- `python/iam/application/orchestration/dag_executor.py`
- `tests/test_orchestration.py`

---

#### Step 15: Observability Module
**Confidence: 85%** | **Priority: P1** | **Risk: Low**
```python
# Test-Driven Development Plan
tests/test_observability.py:
  ✅ test_tracer_captures_calls()
  ✅ test_metrics_collected()
  ✅ test_logging_formatted()

# Implementation Plan
class Tracer:
    """Capture execution traces."""
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
```

**Files to Create**:
- `python/iam/application/observability/tracer.py`
- `python/iam/application/observability/metrics.py`
- `tests/test_observability.py`

---

#### Step 16: Visualization Module
**Confidence: 90%** | **Priority: P2** | **Risk: Low**
```python
# Test-Driven Development Plan
tests/test_visualization.py:
  ✅ test_feature_map_renders()
  ✅ test_attribution_plot_created()
  ✅ test_circuit_graph_drawn()

# Implementation Plan
def plot_feature_map(features: Tensor, **kwargs) -> plt.Figure:
    """Visualize sparse feature activations."""
    fig, ax = plt.subplots()
    ax.imshow(features.to_dense())
    return fig
```

**Files to Create**:
- `python/iam/application/visualization/feature_maps.py`
- `python/iam/application/visualization/attribution_viz.py`
- `tests/test_visualization.py`

---

### Phase 5: Optimization & Integration (Steps 17-20)
**Duration**: 5-7 days  
**Goal**: Performance optimization and production readiness

#### Step 17: CLT Fusion Pass
**Confidence: 50%** | **Priority: P1** | **Risk: High**
```cpp
// Test-Driven Development Plan
mlir/test/Passes/test_clt_fusion.mlir:
  ✅ test_encode_decode_fuses()
  ✅ test_topk_fuses_with_encode()
  ✅ test_performance_improves()

// Implementation Plan
struct CLTFusionPass : public PassWrapper<CLTFusionPass, OperationPass<>> {
  void runOnOperation() override {
    // Pattern: iam.clt.decode(iam.clt.encode(x))
    // Replace: x (if lossless)
    
    // Pattern: iam.clt.topk(iam.clt.encode(x))
    // Replace: iam.clt.encode_topk(x) (fused op)
  }
};
```

**Challenge**: This is the most uncertain part
- Need to deeply understand MLIR pass infrastructure
- Pattern matching can be subtle
- Verification of correctness is critical

**Files to Create**:
- `mlir/lib/Passes/CLTFusionPass.cpp`
- `mlir/include/iam/Passes/CLTFusionPass.hpp`
- `mlir/test/Passes/test_clt_fusion.mlir`

---

#### Step 18: Pythia-170M Integration
**Confidence: 80%** | **Priority: P0** | **Risk: Low**
```python
# Test-Driven Development Plan
notebooks/01_pythia_clt_training.ipynb:
  ✅ Model loads successfully
  ✅ Activations extracted correctly
  ✅ Transcoder trains (10 epochs)
  ✅ Features interpretable

# Implementation Plan
# 1. Load Pythia-170M from HuggingFace
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-170m")

# 2. Hook activations at layer 6
def hook_fn(module, input, output):
    global activations
    activations = output

# 3. Train transcoder
transcoder = Transcoder(d_model=768, d_sae=3072, k=32)
transcoder.train(activations, epochs=10)

# 4. Visualize features
plot_feature_map(transcoder.encode(sample_activations))
```

**Success Criteria**:
- Model loads and runs inference
- Activations captured correctly
- Transcoder trains without errors
- Features show interpretable patterns

**Files to Create**:
- `notebooks/01_pythia_clt_training.ipynb`
- `examples/pythia_integration.py`

---

#### Step 19: End-to-End Pipeline
**Confidence: 70%** | **Priority: P0** | **Risk: Medium**
```python
# Test-Driven Development Plan
notebooks/03_steering_hello_world.ipynb:
  ✅ Full pipeline executes
  ✅ Attribution computed correctly
  ✅ Steering vector applied
  ✅ Output is modified as expected

# Implementation Plan
pipeline = (
    Pipeline()
    .add("extract", lambda x: extract_activations(model, x))
    .add("encode", lambda acts: transcoder.encode(acts))
    .add("attribute", lambda feats: attributor.compute(feats))
    .add("steer", lambda acts: steering.apply(acts, vector))
)

result = pipeline.run({"input": "Hello world"})
```

**Success Criteria**:
- All components work together
- No memory leaks
- Latency <500ms for small model
- Results are reproducible

**Files to Create**:
- `notebooks/03_steering_hello_world.ipynb`
- `examples/end_to_end_demo.py`
- `tests/integration/test_pipeline.py`

---

#### Step 20: Gemmascope-2B Scaling
**Confidence: 45%** | **Priority: P1** | **Risk: High**
```python
# Test-Driven Development Plan
notebooks/04_scaling_gemmascope.ipynb:
  ✅ Memory stays <5GB
  ✅ Latency <500ms per batch
  ✅ Accuracy maintained
  ✅ No memory leaks over time

# Implementation Plan
# This is the ultimate stress test
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Profile memory usage
with memory_profiler():
    with performance_profiler():
        for batch in test_data:
            result = pipeline.run(batch)
            assert memory_usage() < 5 * 1024**3  # 5GB
            assert latency < 0.5  # 500ms
```

**Success Criteria**:
- Memory <5GB (critical)
- Latency <500ms (critical)
- No crashes on long runs
- Results match smaller model

**Files to Create**:
- `notebooks/04_scaling_gemmascope.ipynb`
- `tests/benchmarks/memory_profile.py`
- `tests/benchmarks/perf_clt.py`

---

## 6. Testing Strategy (TDD Approach)

### 6.1 Testing Pyramid
```
           /\
          /  \     E2E Tests (5%)
         /────\    - Full pipeline on real models
        /      \   - Notebooks as integration tests
       /────────\  
      /  Integ   \ Integration Tests (15%)
     /────────────\  
    /   Unit       \ Unit Tests (80%)
   /────────────────\ - Every function tested
  /___________________\ - Fast, isolated, deterministic
```

### 6.2 Test Categories

#### Unit Tests (80% of tests)
```python
# Example: tests/unit/test_transcoder.py
def test_encode_shape_correct():
    """Encode produces correct output shape."""
    transcoder = Transcoder(d_model=768, d_sae=3072, k=32)
    activations = torch.randn(32, 768)  # batch_size=32
    
    features = transcoder.encode(activations)
    
    assert features.shape == (32, 32)  # batch_size, k
    assert features.is_sparse
    assert (features != 0).sum(dim=1).all() == 32  # exactly k active

def test_encode_decode_reconstruction():
    """Encode-decode has bounded reconstruction error."""
    transcoder = Transcoder(d_model=768, d_sae=3072, k=32)
    transcoder.eval()  # Disable dropout
    
    activations = torch.randn(32, 768)
    reconstructed = transcoder.decode(transcoder.encode(activations))
    
    reconstruction_error = torch.norm(activations - reconstructed)
    assert reconstruction_error < 0.1  # Configurable threshold
```

#### Integration Tests (15% of tests)
```python
# Example: tests/integration/test_pipeline.py
def test_clt_attribution_pipeline():
    """CLT + attribution pipeline works end-to-end."""
    model = load_test_model()  # Small test model
    transcoder = Transcoder.load("test_transcoder.pt")
    attributor = IntegratedGradients(model)
    
    # Extract activations
    activations = extract_activations(model, "test input")
    
    # Encode to features
    features = transcoder.encode(activations)
    
    # Compute attribution
    attribution = attributor.attribute(features)
    
    # Assertions
    assert attribution.shape == features.shape
    assert torch.allclose(attribution.sum(), features.sum(), rtol=0.01)
```

#### End-to-End Tests (5% of tests)
```python
# Example: tests/e2e/test_pythia.py
@pytest.mark.slow
def test_pythia_170m_full_pipeline():
    """Full interpretability pipeline on Pythia-170M."""
    # This test is expensive, run nightly
    
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-170m")
    transcoder = train_transcoder(model, num_samples=1000)
    
    # Run complete analysis
    result = analyze_model(
        model=model,
        transcoder=transcoder,
        prompts=["Hello world", "The capital of France is"],
        methods=["attribution", "steering"]
    )
    
    # Check results are sensible
    assert result["attribution"].shape[0] == 2  # 2 prompts
    assert result["steering"]["magnitude"] > 0
```

### 6.3 Test Infrastructure

#### Fixtures
```python
# conftest.py
@pytest.fixture
def small_model():
    """Tiny model for fast testing."""
    return SimpleTransformer(d_model=64, n_layers=2)

@pytest.fixture
def transcoder():
    """Pre-trained transcoder for tests."""
    return Transcoder.load("tests/fixtures/test_transcoder.pt")

@pytest.fixture
def sample_activations():
    """Random activations for quick tests."""
    return torch.randn(32, 768)
```

#### Performance Benchmarks
```python
# tests/benchmarks/perf_clt.py
def test_encode_performance(benchmark):
    """Benchmark CLT encoding speed."""
    transcoder = Transcoder(d_model=768, d_sae=3072, k=32)
    activations = torch.randn(32, 768)
    
    result = benchmark(transcoder.encode, activations)
    
    assert benchmark.stats.mean < 0.05  # <50ms mean
```

#### Memory Profiling
```python
# tests/benchmarks/memory_profile.py
def test_memory_usage_bounded():
    """Ensure memory stays under budget."""
    with memory_tracker() as tracker:
        model = load_large_model()
        transcoder = Transcoder(d_model=2048, d_sae=8192)
        
        for i in range(100):
            activations = extract_activations(model, f"input {i}")
            features = transcoder.encode(activations)
        
        assert tracker.peak_memory < 5 * 1024**3  # <5GB
```

### 6.4 Continuous Integration
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-benchmark
      - name: Run unit tests
        run: pytest tests/unit -v
      - name: Run integration tests
        run: pytest tests/integration -v
      - name: Run benchmarks
        run: pytest tests/benchmarks --benchmark-only
```

---

## 7. Progress Tracking

### 7.1 Daily Progress Log

#### 2025-10-26 (Day 1)
**Status**: Phase 0 - Planning  
**Progress**:
- ✅ Read and understood README.md
- ✅ Analyzed MLIR PoC structure
- ✅ Created comprehensive research plan (this document)

**Blockers**: None  
**Next Steps**: Begin Step 1 (Build System Setup)  
**Confidence Level**: 75% → Unchanged (as expected in planning phase)

---

### 7.2 Sprint Planning

#### Sprint 1 (Days 1-5): Foundation
**Goals**:
- Complete Steps 1-5 (MLIR dialect foundation)
- Have basic IAM operations defined
- All dialect tests passing

**Key Deliverables**:
- [ ] Working build system
- [ ] IAM dialect registers correctly
- [ ] CLT, Attribution, Policy ops defined
- [ ] Round-trip tests pass

**Success Metrics**:
- All unit tests pass
- Documentation for each component
- Code review completed

---

#### Sprint 2 (Days 6-12): C++ Core
**Goals**:
- Complete Steps 6-10 (C++ implementation)
- Have performant C++ backend
- Python bindings working

**Key Deliverables**:
- [ ] Tensor and memory management
- [ ] CLT transcoder implementation
- [ ] Attribution implementation
- [ ] Steering implementation
- [ ] PyBind11 bindings

**Success Metrics**:
- Benchmark: Encode <50ms
- Memory: No leaks detected
- Python: Can import and use

---

#### Sprint 3 (Days 13-18): Python API
**Goals**:
- Complete Steps 11-13 (Python layer)
- Match SAELens/Captum API
- Have working examples

**Key Deliverables**:
- [ ] Python CLT module
- [ ] Python attribution module
- [ ] Python steering module
- [ ] Integration tests pass

**Success Metrics**:
- API matches SAELens ergonomics
- Examples run successfully
- Documentation complete

---

#### Sprint 4 (Days 19-25): Integration
**Goals**:
- Complete Steps 14-20 (full system)
- Demonstrate on Pythia-170M
- Scale to larger models

**Key Deliverables**:
- [ ] Orchestration layer
- [ ] Observability/visualization
- [ ] Pythia integration notebook
- [ ] Scaling tests

**Success Metrics**:
- End-to-end pipeline works
- Memory <5GB for Gemmascope-2B
- Latency <500ms

---

### 7.3 Milestone Tracking

| Milestone | Target Date | Status | Confidence |
|-----------|-------------|--------|-----------|
| Build System Working | Day 2 | 🔲 Not Started | 85% |
| IAM Dialect Defined | Day 5 | 🔲 Not Started | 70% |
| C++ Core Complete | Day 12 | 🔲 Not Started | 75% |
| Python API Complete | Day 18 | 🔲 Not Started | 80% |
| Pythia Integration | Day 22 | 🔲 Not Started | 80% |
| Scaling Validated | Day 25 | 🔲 Not Started | 45% |

---

## 8. Risk Analysis & Mitigation

### 8.1 Technical Risks

#### Risk 1: MLIR Dialect Complexity
**Probability**: Medium (40%)  
**Impact**: High  
**Description**: Defining custom MLIR operations might be more complex than anticipated, especially type checking and verification.

**Mitigation Strategy**:
1. Start with simplest possible operations
2. Study existing dialects thoroughly (func, arith, linalg)
3. Use MLIR toy tutorial as guide
4. Ask for help on MLIR Discord if stuck
5. Budget extra time for learning (already in plan)

**Fallback Plan**:
- If MLIR dialect too complex, use MLIR only for lowering
- Keep high-level logic in Python
- Still get some performance benefits

**Confidence Impact**: 70% → 60% if this materializes

---

#### Risk 2: Performance Targets Not Met
**Probability**: Medium (50%)  
**Impact**: Medium  
**Description**: Might not achieve 5-10x speedup over SAELens initially.

**Mitigation Strategy**:
1. Profile early and often
2. Identify bottlenecks with real data
3. Focus on most expensive operations first
4. Use SIMD/GPU when beneficial
5. Accept iterative improvement

**Fallback Plan**:
- Start with 2-3x speedup as MVP
- Document optimization opportunities for future
- User value from unified API even without max performance

**Confidence Impact**: Performance targets are stretch goals, so doesn't affect core hypothesis

---

#### Risk 3: Memory Leaks in C++
**Probability**: Low (20%)  
**Impact**: High  
**Description**: C++ memory management bugs could cause production issues.

**Mitigation Strategy**:
1. Use RAII extensively
2. Smart pointers everywhere
3. Memory profiling on every commit
4. Valgrind in CI pipeline
5. Leak detection tools

**Fallback Plan**:
- Rely more on Python garbage collection
- Use memory pooling to reduce allocations
- Profile and fix leaks as they appear

**Confidence Impact**: 75% → 60% if leaks are persistent

---

#### Risk 4: PyBind11 Integration Issues
**Probability**: Low (15%)  
**Impact**: Low  
**Description**: Python-C++ integration might have subtle bugs.

**Mitigation Strategy**:
1. Extensive unit tests for bindings
2. Test NumPy integration thoroughly
3. Handle Python exceptions correctly
4. Use py::array_t for zero-copy

**Fallback Plan**:
- Use pybind11_numpy for easier NumPy integration
- Simplify API if needed
- Copy data if zero-copy proves difficult

**Confidence Impact**: Minimal - PyBind11 is mature and well-documented

---

#### Risk 5: Scaling to Gemmascope-2B Fails
**Probability**: Medium-High (60%)  
**Impact**: Medium  
**Description**: Memory and performance targets for large model might not be achievable in MVP timeframe.

**Mitigation Strategy**:
1. Profile incrementally (170M → 410M → 1B → 2B)
2. Identify memory bottlenecks early
3. Use streaming/chunking if needed
4. Optimize critical path first

**Fallback Plan**:
- Demonstrate on smaller models (170M, 410M)
- Document scaling as future work
- Show clear path to improvement
- Focus on correctness over scale for MVP

**Confidence Impact**: Already reflected in 45% confidence for Step 20

---

### 8.2 Project Risks

#### Risk 6: Scope Creep
**Probability**: Medium (35%)  
**Impact**: High  
**Description**: Temptation to add features beyond MVP scope.

**Mitigation Strategy**:
1. Strict adherence to 20-step plan
2. Track scope changes explicitly
3. "YAGNI" principle - You Aren't Gonna Need It
4. Focus on core functionality first

**Fallback Plan**:
- Maintain "Future Work" document
- Defer non-critical features
- Prioritize ruthlessly

---

#### Risk 7: Time Estimation Errors
**Probability**: High (70%)  
**Impact**: Medium  
**Description**: Steps might take longer than estimated.

**Mitigation Strategy**:
1. Track actual time spent
2. Adjust future estimates based on reality
3. Buffer time in schedule (25% contingency)
4. Daily progress reviews

**Fallback Plan**:
- Cut scope if needed
- Deliver MVP with reduced features
- Document clearly what's complete vs. pending

---

### 8.3 Risk Matrix
```
Impact
  ^
H │  ■ MLIR Complexity      □ Memory Leaks
  │
M │  □ Performance          ■ Scaling
  │  □ Time Estimation
  │
L │                         □ PyBind11
  │  
  └─────────────────────────────> Probability
    Low    Medium    High

Legend:
■ = Active mitigation in place
□ = Monitor but accept risk
```

---

## 9. Decision Log

### Decision 1: Tensor Library Choice
**Date**: 2025-10-26  
**Decision**: Use Eigen for initial implementation  
**Alternatives Considered**: xtensor, custom implementation  
**Rationale**:
- Eigen is mature and well-documented
- Good SIMD support
- Easy PyBind11 integration
- Can switch to xtensor later if needed

**Impact**: Step 6 (Core Types & Memory)  
**Reversibility**: Medium (could switch libraries)

---

### Decision 2: Test-First Development
**Date**: 2025-10-26  
**Decision**: Write tests before implementation  
**Alternatives Considered**: Test-after, no formal testing  
**Rationale**:
- Catches bugs early
- Documents expected behavior
- Guides implementation
- Prevents regression

**Impact**: All steps  
**Reversibility**: Low (established practice)

---

### Decision 3: Progressive Lowering Strategy
**Date**: 2025-10-26  
**Decision**: IAM dialect → StableHLO → XLA  
**Alternatives Considered**: Direct LLVM lowering  
**Rationale**:
- StableHLO is designed for ML workloads
- Enables TPU backend
- Better community support
- Handles dynamic shapes well

**Impact**: Step 17 (Optimization passes)  
**Reversibility**: Low (architecture choice)

---

### Decision 4: MVP Scope - 20 Steps
**Date**: 2025-10-26  
**Decision**: Follow README's 20-step plan exactly  
**Alternatives Considered**: Simpler/more ambitious scopes  
**Rationale**:
- Well thought-out structure
- Clear deliverables
- Achievable in timeframe
- Demonstrates all key capabilities

**Impact**: Overall project structure  
**Reversibility**: Medium (can adjust scope)

---

### Decision 5: Python API Matches SAELens
**Date**: 2025-10-26  
**Decision**: Make API as similar to SAELens as possible  
**Alternatives Considered**: Novel API design  
**Rationale**:
- Reduces adoption friction
- Leverages existing knowledge
- Easier to compare
- Familiar patterns

**Impact**: Step 11 (CLT Python Module)  
**Reversibility**: High (API can evolve)

---

## 10. Learning Resources

### 10.1 MLIR Resources

#### Essential Reading
1. **MLIR Toy Tutorial** (7 chapters)
   - URL: https://mlir.llvm.org/docs/Tutorials/Toy/
   - Priority: P0 - Must complete
   - Time: 2-3 days

2. **MLIR Language Reference**
   - URL: https://mlir.llvm.org/docs/LangRef/
   - Priority: P0 - Reference material
   - Keep open while coding

3. **Defining Dialects (ODS)**
   - URL: https://mlir.llvm.org/docs/DefiningDialects/Operations/
   - Priority: P0 - For Steps 2-5
   - Time: 1 day

4. **Pattern Rewriting**
   - URL: https://mlir.llvm.org/docs/PatternRewriter/
   - Priority: P1 - For Step 17
   - Time: 0.5 day

#### MLIR Examples
- Study existing dialects:
  - `func` - Function operations
  - `arith` - Arithmetic operations
  - `linalg` - Linear algebra operations
  - `tensor` - Tensor operations

### 10.2 C++ Resources

#### Memory Management
1. **Effective Modern C++** (Scott Meyers)
   - Smart pointers
   - RAII patterns
   - Move semantics

2. **C++ Core Guidelines**
   - URL: https://isocpp.github.io/CppCoreGuidelines/
   - Reference for best practices

#### Performance Optimization
1. **Eigen Documentation**
   - URL: https://eigen.tuxfamily.org/
   - Matrix operations
   - SIMD vectorization

2. **Valgrind Tutorial**
   - Memory profiling
   - Leak detection

### 10.3 Python Resources

#### PyBind11
1. **PyBind11 Documentation**
   - URL: https://pybind11.readthedocs.io/
   - Python-C++ binding
   - NumPy integration

2. **PyBind11 Examples**
   - Study mature projects using PyBind11
   - PyTorch, TensorFlow bindings

#### Testing
1. **Pytest Documentation**
   - Fixtures
   - Parametrization
   - Benchmarking

### 10.4 Interpretability Research

#### Key Papers
1. **Toy Models of Superposition** (Anthropic)
   - Understanding feature geometry
   - Superposition hypothesis

2. **Scaling Monosemanticity** (Anthropic)
   - Large-scale SAE training
   - Feature interpretation

3. **Circuit Thread** (Anthropic)
   - Circuit discovery
   - Mechanistic interpretability

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| CLT | Contrastive Latent Transcoder - generalized sparse auto-encoder |
| IAM | Interpretable AI with MLIR - this project |
| MLIR | Multi-Level Intermediate Representation - compiler framework |
| SAE | Sparse Auto-Encoder |
| StableHLO | High-Level Operations - ML-focused IR |
| TableGen | MLIR's operation definition language |
| TopK | Select top K values (sparsity inducing) |
| XLA | Accelerated Linear Algebra - Google's compiler |

---

## Appendix B: File Structure Template
```
iam/
├── CMakeLists.txt                    # Build configuration
├── pyproject.toml                    # Python package metadata
├── setup.py                          # Installation script
├── RESEARCH_PLAN.md                  # This document
├── README.md                         # Project overview
│
├── cpp/                              # C++ implementation
│   ├── include/iam/
│   │   ├── core/
│   │   │   ├── Tensor.hpp
│   │   │   ├── Memory.hpp
│   │   │   └── Types.hpp
│   │   ├── clt/
│   │   │   ├── Transcoder.hpp
│   │   │   ├── Encoder.hpp
│   │   │   ├── Decoder.hpp
│   │   │   └── TopKActivation.hpp
│   │   ├── attribution/
│   │   │   ├── IntegratedGradients.hpp
│   │   │   ├── FeatureSelector.hpp
│   │   │   └── CircuitExtractor.hpp
│   │   └── steering/
│   │       ├── VectorConstructor.hpp
│   │       ├── PolicyApplier.hpp
│   │       └── SafetyVerifier.hpp
│   │
│   └── src/
│       ├── core/
│       ├── clt/
│       ├── attribution/
│       └── steering/
│
├── mlir/                             # MLIR dialect definition
│   ├── include/iam/
│   │   ├── Dialect/
│   │   │   ├── IAMDialect.hpp
│   │   │   ├── CLTOps.hpp
│   │   │   ├── AttributionOps.hpp
│   │   │   └── PolicyOps.hpp
│   │   └── Passes/
│   │       ├── CLTFusionPass.hpp
│   │       ├── AttributionCachePass.hpp
│   │       └── PolicyProjectionPass.hpp
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
│   ├── tablegen/
│   │   ├── IAMBase.td
│   │   ├── CLTOps.td
│   │   ├── AttributionOps.td
│   │   └── PolicyOps.td
│   │
│   └── test/
│       ├── Dialect/
│       │   └── IAM/
│       │       ├── test_base.mlir
│       │       ├── test_clt_ops.mlir
│       │       ├── test_attr_ops.mlir
│       │       └── test_policy_ops.mlir
│       └── Passes/
│           └── test_clt_fusion.mlir
│
├── python/                           # Python API
│   ├── iam/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── types.py
│   │   ├── vector/
│   │   │   ├── clt/
│   │   │   │   ├── transcoder.py
│   │   │   │   ├── activation.py
│   │   │   │   └── training.py
│   │   │   ├── attribution/
│   │   │   │   ├── integrated_gradients.py
│   │   │   │   ├── feature_selector.py
│   │   │   │   └── circuit_extractor.py
│   │   │   └── steering/
│   │   │       ├── vector_constructor.py
│   │   │       ├── policy_applier.py
│   │   │       └── safety_verifier.py
│   │   └── application/
│   │       ├── orchestration/
│   │       │   ├── pipeline.py
│   │       │   └── dag_executor.py
│   │       ├── observability/
│   │       │   ├── tracer.py
│   │       │   └── metrics.py
│   │       └── visualization/
│   │           ├── feature_maps.py
│   │           └── attribution_viz.py
│   │
│   └── bindings/                     # PyBind11 bindings
│       ├── clt_bindings.cpp
│       ├── attribution_bindings.cpp
│       └── steering_bindings.cpp
│
├── tests/                            # Test suite
│   ├── unit/
│   │   ├── test_tensor.cpp
│   │   ├── test_clt.cpp
│   │   ├── test_attribution.cpp
│   │   ├── test_steering.cpp
│   │   ├── test_clt_api.py
│   │   ├── test_attribution_api.py
│   │   └── test_steering_api.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_mlir_lowering.py
│   ├── benchmarks/
│   │   ├── perf_clt.py
│   │   └── memory_profile.py
│   └── fixtures/
│       └── test_transcoder.pt
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_pythia_clt_training.ipynb
│   ├── 02_attribution_demo.ipynb
│   ├── 03_steering_hello_world.ipynb
│   └── 04_scaling_gemmascope.ipynb
│
└── examples/                         # Usage examples
    ├── pythia_integration.py
    ├── end_to_end_demo.py
    └── custom_analysis.py
```

---

## Appendix C: Confidence Calibration

### How to Update Confidence Levels

Confidence levels should be updated based on:
1. **Test Results**: Passing tests → +5-10%
2. **Benchmarks**: Meeting targets → +10-15%
3. **Integration**: Components work together → +5%
4. **Bugs Found**: Serious bugs → -10-20%
5. **Blocked**: Can't proceed → -20-30%

### Confidence Interpretation

| Range | Meaning | Action |
|-------|---------|--------|
| 90-100% | Very confident | Proceed, minimal risk |
| 75-89% | Confident | Proceed, monitor closely |
| 60-74% | Moderate | Proceed with caution |
| 40-59% | Uncertain | Investigate before proceeding |
| 0-39% | Low confidence | Seek help, consider pivot |

### Example Confidence Evolution
```
Step 6: Core Types & Memory
Initial: 80% (Have C++ experience, clear requirements)
After design review: 82% (+2%, design validated)
After unit tests pass: 88% (+6%, implementation working)
After memory profiling: 85% (-3%, found minor leak)
After leak fixed: 90% (+5%, all tests passing)
```

---

## Appendix D: Next Actions

### Immediate (This Week)
1. [ ] Complete Phase 0 planning (this document)
2. [ ] Search project knowledge for requirements
3. [ ] Set up development environment
4. [ ] Begin Step 1: Build System Setup

### Short-term (Next 2 Weeks)
1. [ ] Complete Phase 1 (Foundation)
2. [ ] Begin Phase 2 (C++ Core)
3. [ ] Weekly progress reviews
4. [ ] Update confidence levels

### Medium-term (Next Month)
1. [ ] Complete Phase 2 (C++ Core)
2. [ ] Complete Phase 3 (Python API)
3. [ ] Begin Phase 4 (Application Layer)
4. [ ] Pythia-170M integration

### Long-term (2+ Months)
1. [ ] Complete Phase 5 (Optimization)
2. [ ] Scale to Gemmascope-2B
3. [ ] Write paper
4. [ ] Release open-source

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-26 | 0.1 | Initial research plan created | Claude |
| | | | |
| | | | |

---

**End of Research Plan v0.1**

This is a living document. Update regularly as:
- Requirements evolve
- Risks materialize
- Confidence changes
- Decisions are made
- Progress is achieved

Remember: **Failing fast is better than failing slowly. Test early, test often, calibrate confidence continuously.**