# FPGA_Excellarator: HW/SW Co-Design CNN Accelerator

This repository presents the design, implementation, and rigorous optimization of a hardware accelerator for Convolutional Neural Networks (CNNs), synthesized on an Intel/Altera DE10-Lite FPGA and controlled by a custom RISC-V (Kuntz5) CPU.

---

## 🎯 Project Objective & Performance Breakthrough

The ultimate goal of this project was to offload compute-heavy deep learning workloads from the processor to a dedicated hardware execution engine, minimizing latency and maximizing throughput.

* **The Baseline (Software Only):** When running the inference algorithm purely in software on the RISC-V (Kuntz5) core, the execution was extraordinarily slow, requiring **over 1,000,000 clock cycles** to process a single operational pass due to the sequential nature of scalar ALU operations.
* **The Solution (Hardware Accelerated):** By designing a custom hardware accelerator architecture and applying aggressive co-design refactoring, we smashed the processing bottleneck. We achieved an astonishing **>250x overall speedup**, successfully dropping the end-to-end execution latency from the initial hardware setup of 8,337 cycles down to exactly **4,029 clock cycles**—surpassing our target benchmark of 4,000 cycles.

---

## 🗺️ System Architecture

The accelerator subsystem (`slrx.sv`) connects directly to the CPU's memory bus via an optimized multiplexer arbiter (`xmem_intrf_mux.sv`). The accelerator pipelines and parallelizes three core neural network operations:
1. **Convolution (CONV):** Spatial feature extraction utilizing parallel multiply-accumulate engines.
2. **Pooling (POOL):** Max-pooling spatial downsampling to reduce feature map dimensionality.
3. **Linear / Fully-Connected (LINEAR):** Fully parallel matrix-vector dot-product calculations with inline rectified linear activation (ReLU).

![System Architectural Block Diagram](pictures/system_architecture.png)

---

## 🏎️ Handshake Optimization & Evolution Timeline

To lower the cycle count below our target, we systematically re-engineered the communication protocol (handshake) and state machines between the software driver layer (C) and the hardware state machines (RTL). Here is the breakdown of what was achieved at each developmental stage:

### Step 1: The Baseline Hardware Architecture (Output-Stationary Loop)
In the first accelerated iteration, the hardware followed a restrictive *Output-Stationary* loop dataflow. The software driver launched operations sequentially, computing only two output columns per hardware invocation (Block A and Block B).

* **What we did:** We implemented basic control registers to offload core math functions.
* **The Bottleneck:** The processor had to stay trapped in a heavy `while(!HOST_REG(XLR_DONE_RI))` polling loop, reloading the input vector from XMEM over and over for every single pair of output elements, wasting thousands of cycles on bus overhead.

![Baseline Waveform Timeline - 8337 Cycles](pictures/cycles_for_operation.png)

### Step 2: Re-Architecting to a Broadcast-MAC Array (Input-Stationary)
We completely re-architected the execution model into an *Input-Stationary / Broadcast-MAC Array* to prioritize extreme data reuse.

* **What we did in Software (C):** Modified the driver to load and lock the input feature vector into internal hardware registers *once* during a unified `LIN_SETUP` phase. We also transposed the weight matrix layout into an *Input-Major* scheme.
* **What we did in Hardware (RTL):** Formed a physical array of 32 parallel signed 32-bit accumulators (`acc`). In the newly introduced `STREAM_W` state, the hardware broadcasted a single input activation to all 32 accumulators simultaneously while streaming a continuous row of packed weights across a wide 256-bit memory transaction every cycle.

![Broadcast-MAC Phase 2 Waveform](pictures/cycles_for_op_2.png)

### Step 3: Padded Fixed 32-Byte Stride Integration (The Final Leap to 4,029 Cycles)
While Phase 2 significantly mitigated memory bandwidth constraints, layers with asymmetric dimensions (e.g., $FC1$ with an output dimension of 27) introduced dynamic control bubbles and simulator unknown bits (`X` values in Xcelium) due to unaligned memory streams.

* **What we did in Software (C):** Integrated high-speed `memset` blocks to completely clean and pad the transposed weight matrix (`lin_w_trn_t`) and biases (`lin_b_padded`) to static 32-byte boundaries in XMEM, preventing uninitialized memory leaks.
* **What we did in Hardware (RTL):** We decoupled the FSM memory requests from dynamic layer boundaries. The RTL was refactored to stream strict 32-byte chunks continuously (`mem_size_bytes = DIM_MAX_SIZE`), using the dynamic bounds (`lin_arr_out_dim`) solely as a mask during accumulation and activation write-back.

This modification fully saturated the 32-byte memory bus, eliminated protocol bubbles, and successfully brought down the end-to-end execution latency to **4,029 cycles**.

![Final Optimized Waveform - 4029 Cycles](pictures/4029_cycles.png)

---

## 📂 Project Repository Structure

```text
├── hw/
│   └── xlrs/slrx/
│       ├── top/
│       │   ├── slrx.sv            # Top-level accelerator coordinator
│       │   ├── xmem_intrf_mux.sv  # Shared 32-byte XMEM arbiter MUX
│       │   └── slrx_regs_intrf.sv # Host register interface logic
│       ├── conv/
│       │   └── conv.sv            # Convolution engine state machine
│       ├── pool/
│       │   └── pool.sv            # Max-pooling hardware module
│       └── linear/
│           └── linear.sv          # Padded Broadcast-MAC array module
├── sw/
│   └── apps/slrx/
│       ├── slrx.c                 # Top-level inference controller
│       ├── conv.c / pool.c        # Hardware driver routines
│       └── linear.c               # Stride-padded weight transposition driver
└── pictures/
    └── *.png                      # Waveform simulations and architecture diagrams
