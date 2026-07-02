# FPGA_Excellarator: HW/SW Co-Design CNN Accelerator

This repository presents the design, implementation, and rigorous optimization of a hardware accelerator for Convolutional Neural Networks (CNNs), synthesized on an Intel/Altera DE10-Lite FPGA and controlled by a custom RISC-V (Kuntz5) CPU. 

By leveraging hardware/software co-design methodologies, we refactored the execution pipeline and register handshakes, achieving a massive **>2x performance speedup**—slashing the end-to-end simulation latency from **8,337 cycles down to 4,029 clock cycles**.

---

## 🗺️ System Architecture

The accelerator subsystem (`slrx.sv`) connects directly to the CPU's memory bus via an optimized multiplexer arbiter (`xmem_intrf_mux.sv`). The accelerator pipelines and parallelizes three core neural network operations:
1. **Convolution (CONV):** Spatial feature extraction utilizing parallel multiply-accumulate engines.
2. **Pooling (POOL):** Max-pooling spatial downsampling to reduce feature map dimensionality.
3. **Linear / Fully-Connected (LINEAR):** Fully parallel matrix-vector dot-product calculations with inline rectified linear activation (ReLU).

![System Architectural Block Diagram](pictures/system_architecture.png)

---

## 🏎️ Handshake Optimization & Evolution Timeline

The primary objective of this project was to systematically profile and eliminate execution stalls (dead cycles) during command issue, metadata setup, and memory writing phases between the software driver layer (C) and the hardware state machines (RTL).

### Phase 1: Baseline Architecture (Output-Stationary)
In the initial baseline implementation, the hardware followed an *Output-Stationary* loop dataflow. The software driver launched operations sequentially, computing only two output columns per hardware invocation (Block A and Block B). 

- **The Bottleneck:** The CPU spent hundreds of cycles polling the `XLR_DONE_RI` register, forcing redundant input vector loads from XMEM for every pair of output elements, severely starving the computation pipeline.

![Baseline Waveform Timeline - 8337 Cycles](pictures/cycles_for_operation.png)

### Phase 2: Broadcast-MAC Refactoring (Input-Stationary)
We completely re-architected the execution model into an *Input-Stationary / Broadcast-MAC Array* to prioritize data reuse.
- **HW Realignment:** Implemented a physical array of 32 parallel signed 32-bit accumulators (`acc`).
- **Data Reuse:** The C driver fetches and locks the input feature vector into internal hardware registers *once* during a unified `LIN_SETUP` phase.
- **Weight Streaming:** In the `STREAM_W` state, the hardware broadcasts a single input activation to all 32 accumulators while streaming a packed row of weights across a wide 256-bit memory transaction every cycle.

![Broadcast-MAC Phase 2 Waveform](pictures/cycles_for_op_2.png)

### Phase 3: Fixed 32-Byte Stride Version (Final Success)
While Phase 2 mitigated memory bandwidth constraints, layers with asymmetric dimensions (e.g., $FC1$ with an output dimension of 27) introduced dynamic control bubbles and simulator unknown bits (`X` values in Xcelium) due to unaligned memory packages.

- **The Vision:** We decoupled the FSM memory requests from the dynamic layer boundaries, binding both hardware streams and software buffers to a fixed, rigid **32-byte physical memory stride (`DIM_MAX_SIZE`)**.
- **Software Buffer Padding:** The C driver utilizes high-speed `memset` blocks to completely initialize and pad the transposed weight matrix (`lin_w_trn_t`) and biases (`lin_b_padded`) to 32-byte boundaries in XMEM, preventing uninitialized memory leaks.
- **Hardware Masking:** The RTL streams strict 32-byte chunks continuously, using the dynamic bounds (`lin_arr_out_dim`) solely as a mask during accumulation and activation write-back.

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
│       │   └── sl
