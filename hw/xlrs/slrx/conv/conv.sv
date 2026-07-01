//=============================================================================
// File: conv.sv
// Description: Convolution Accelerator (5x5) - DUAL parallel compute blocks
//
//   Contract: ONE CONV_WINDOW command computes the ENTIRE output feature-map.
//     The FSM internally loops row-pair index 0..half_rows-1; for each
//     row-pair:
//     - Block A : output row  rA = row_pair_idx              (top half)
//     - Block B : output row  rB = row_pair_idx + half_rows  (bottom half)
//   half_rows = ceil(out_dim/2). The C driver issues CONV_SETUP once and
//   CONV_WINDOW once per layer (no per-row-pair host round trip) -- the FSM
//   loops STREAM -> LOAD_A directly between row-pairs with zero bubble
//   cycles instead of returning to IDLE/DONE for a fresh host command.
//
//   Each block has its own 5-row input buffer. The 25-tap MAC (5x5 kernel dot
//   product) is split into a 2-STAGE PIPELINE to shorten the combinational
//   critical path (STA showed the single-cycle 25-term accumulation was the
//   Fmax-limiting path at ~42MHz for the full chip):
//     Stage 1 (same cycle as the window fetch): sum the first 13 taps into
//              partial1, the remaining 12 taps into partial2 -- roughly half
//              the adder-chain depth of the original all-in-one-cycle sum --
//              then register both partial sums.
//     Stage 2 (next cycle): combine partial1 + partial2 + bias (one cheap
//              add) then ReLU/descale/saturate -- a short combinational tail.
//   This is a TRUE overlapping pipeline (stage 1 fetches column N+1 while
//   stage 2 finalizes column N), so steady-state throughput is still one
//   column per cycle -- only 1 extra cycle of pipeline fill/drain latency
//   is added per row-pair, not per column. The write port and its
//   round-robin arbiter are completely unchanged.
//
//   Arithmetic (bias + sum(pixel*weight), ReLU, descale >>>8, saturate) is
//   IDENTICAL to the proven single-block version -- only WHEN the sum is
//   totalled (over 2 cycles instead of 1) changed, not the math itself.
//=============================================================================

import xbox_def_pkg::*;
import slrx_def_pkg::*;

module conv (
  input   clk,
  input   rst_n,

  slrx_regs_intrf.xlr slrx_regs_intrf,

  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  //===========================================================================
  // State Machine
  //===========================================================================
  enum {
     IDLE,
     READ_KERNEL,
     LOAD_A,        // load 5 input rows for block A's region
     LOAD_B,        // load 5 input rows for block B's region
     STREAM,        // both blocks scan their rows in parallel
     DONE
  } next_state, state;

  //===========================================================================
  // Local Parameters
  //===========================================================================
  localparam DIM_MAX_SIZE      = 32;
  localparam KERNEL_DIM        = 5;
  localparam KERNEL_SIZE       = KERNEL_DIM*KERNEL_DIM;
  localparam MAX_DOT_PROD_WIDTH= 16+$clog2(KERNEL_SIZE);
  localparam ARR_IDX_W         = $clog2(DIM_MAX_SIZE);
  localparam PARTIAL_SPLIT     = 13; // first PARTIAL_SPLIT taps -> partial1, rest -> partial2

  //===========================================================================
  // Control
  //===========================================================================
  logic conv_start;
  logic conv_done;
  logic clear_done_on_read;
  logic conv_active;

  //===========================================================================
  // Kernel + two input row buffers (one per block)
  //===========================================================================
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel, kernel_ps;

  logic [KERNEL_DIM-1:0][DIM_MAX_SIZE-1:0][7:0] bufA, bufA_ps;
  logic [KERNEL_DIM-1:0][DIM_MAX_SIZE-1:0][7:0] bufB, bufB_ps;

  //===========================================================================
  // Host-configured registers
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] conv_kernel_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_in_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_out_addr;

  logic [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;

  logic [ARR_IDX_W:0] conv_arr_in_dim;
  logic [ARR_IDX_W:0] conv_arr_out_dim;

  logic [ARR_IDX_W:0] half_rows;          // ceil(out_dim/2) = number of row-pair iterations
  logic [ARR_IDX_W:0] row_b;              // current row-pair's block B row = out_row_cnt + half_rows

  // Internal row-pair loop counter: ONE CONV_WINDOW command now processes ALL row-pairs
  // (0..half_rows-1) inside the FSM, looping STREAM -> LOAD_A directly (zero bubble
  // cycles) instead of returning to the host for a per-row-pair command.
  // OUT_ROW_IDX_RI is no longer read by this module.
  logic [ARR_IDX_W:0] out_row_cnt, out_row_cnt_ps;

  //===========================================================================
  // Row read addressing (shared by the two load phases)
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr, arr_in_row_addr_ps;
  logic [ARR_IDX_W-1:0]       buf_load_row_idx, buf_load_row_idx_ps;
  logic                       is_last_load_row;

  //===========================================================================
  // Per-block STREAM datapath
  //===========================================================================
  logic [ARR_IDX_W:0]         colA, colA_ps;     // compute column (block A) -- stage 1 fetch pointer
  logic [ARR_IDX_W:0]         colB, colB_ps;     // compute column (block B) -- stage 1 fetch pointer
  logic [XMEM_ADDR_WIDTH-1:0] baseA, baseA_ps;   // output base addr (row A)
  logic [XMEM_ADDR_WIDTH-1:0] baseB, baseB_ps;   // output base addr (row B)

  // Stage 1 -> Stage 2 pipeline registers (the new pipeline stage): holds the
  // two partial sums for one column until stage 2 can finalize+forward them.
  logic signed [MAX_DOT_PROD_WIDTH-1:0] p1A, p1A_ps, p2A, p2A_ps;
  logic signed [MAX_DOT_PROD_WIDTH-1:0] p1B, p1B_ps, p2B, p2B_ps;
  logic [ARR_IDX_W:0]                   p_colA, p_colA_ps, p_colB, p_colB_ps;
  logic                                 p_validA, p_validA_ps, p_validB, p_validB_ps;

  logic                wr_validA, wr_validA_ps;  // output pipeline reg A (stage 2 -> write-hold)
  logic [7:0]          wr_byteA,  wr_byteA_ps;
  logic [ARR_IDX_W:0]  wr_colA,   wr_colA_ps;

  logic                wr_validB, wr_validB_ps;  // output pipeline reg B (stage 2 -> write-hold)
  logic [7:0]          wr_byteB,  wr_byteB_ps;
  logic [ARR_IDX_W:0]  wr_colB,   wr_colB_ps;

  logic                activeB, activeB_ps;      // block B has a valid row this call
  logic                last_served, last_served_ps; // 0=A served last, 1=B served last

  //===========================================================================
  // Two parallel 25-MAC blocks, split into 2 pipeline stages:
  //   Stage 1 (combinational, this cycle's colA/colB): partial1/partial2
  //   Stage 2 (combinational, from LAST cycle's registered p1/p2): final byte
  //===========================================================================
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] winA, winB;
  logic signed [MAX_DOT_PROD_WIDTH-1:0]       partial1A, partial2A, partial1B, partial2B;
  logic [7:0]                                 resultA_final, resultB_final;

  always_comb begin
    for (int i = 0; i < KERNEL_DIM; i++)
      for (int j = 0; j < KERNEL_DIM; j++)
        winA[i][j] = bufA[i][colA + j];
    partial1A = calc_conv_partial_sum(kernel, winA, 0, PARTIAL_SPLIT);
    partial2A = calc_conv_partial_sum(kernel, winA, PARTIAL_SPLIT, KERNEL_SIZE);
    resultA_final = calc_conv_finalize(p1A, p2A, conv_bias_val);

    for (int i = 0; i < KERNEL_DIM; i++)
      for (int j = 0; j < KERNEL_DIM; j++)
        winB[i][j] = bufB[i][colB + j];
    partial1B = calc_conv_partial_sum(kernel, winB, 0, PARTIAL_SPLIT);
    partial2B = calc_conv_partial_sum(kernel, winB, PARTIAL_SPLIT, KERNEL_SIZE);
    resultB_final = calc_conv_finalize(p1B, p2B, conv_bias_val);
  end

  //===========================================================================
  // Host Register Interface (unchanged)
  //===========================================================================
  assign slrx_regs_intrf.xlr_done = conv_done;

  assign slrx_cmd = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);

  assign conv_active = (slrx_cmd == CONV_SETUP) || (slrx_cmd == CONV_WINDOW);
  assign conv_start  = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && conv_active;
  assign clear_done_on_read = conv_active && slrx_regs_intrf.xlr_done_ack;

  assign conv_kernel_addr  = slrx_regs_intrf.host_regs[WGT_ADDR_RI];
  assign conv_bias_val     = $signed(slrx_regs_intrf.host_regs[CONV_BIAS_VAL_RI][MAX_DOT_PROD_WIDTH-1:0]);
  assign conv_arr_in_addr  = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];
  assign conv_arr_out_addr = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI];
  assign conv_arr_in_dim   = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];

  assign conv_arr_out_dim  = conv_arr_in_dim - (KERNEL_DIM - 1);
  assign half_rows         = (conv_arr_out_dim + 1) >> 1;          // ceil(out_dim/2) = row-pair iterations
  assign row_b             = out_row_cnt + half_rows;              // current row-pair's block B row
  assign is_last_load_row  = (buf_load_row_idx == (KERNEL_DIM - 1));

  //===========================================================================
  // FSM + datapath - Combinational
  //===========================================================================
  logic pendingA, pendingB, grantA, grantB;
  logic ackA, ackB, slot_freeA, slot_freeB;
  logic slot_free_pA, slot_free_pB;
  logic doneA, doneB;

  always_comb begin

    next_state = state;

    mem_intf_read.mem_size_bytes  = 0;
    mem_intf_read.mem_start_addr  = 0;
    mem_intf_read.mem_req         = 0;

    mem_intf_write.mem_size_bytes = 1;
    mem_intf_write.mem_data       = wr_byteA;
    mem_intf_write.mem_start_addr = baseA + wr_colA;
    mem_intf_write.mem_req        = 0;

    conv_done = 0;

    // hold registers
    bufA_ps = bufA;  bufB_ps = bufB;  kernel_ps = kernel;
    arr_in_row_addr_ps  = arr_in_row_addr;
    buf_load_row_idx_ps = buf_load_row_idx;
    colA_ps = colA;  colB_ps = colB;
    baseA_ps = baseA; baseB_ps = baseB;
    p1A_ps = p1A; p2A_ps = p2A; p_colA_ps = p_colA; p_validA_ps = p_validA;
    p1B_ps = p1B; p2B_ps = p2B; p_colB_ps = p_colB; p_validB_ps = p_validB;
    wr_validA_ps = wr_validA; wr_byteA_ps = wr_byteA; wr_colA_ps = wr_colA;
    wr_validB_ps = wr_validB; wr_byteB_ps = wr_byteB; wr_colB_ps = wr_colB;
    activeB_ps = activeB; last_served_ps = last_served;
    out_row_cnt_ps = out_row_cnt;

    pendingA = 1'b0; pendingB = 1'b0; grantA = 1'b0; grantB = 1'b0;
    ackA = 1'b0; ackB = 1'b0; slot_freeA = 1'b0; slot_freeB = 1'b0;
    slot_free_pA = 1'b0; slot_free_pB = 1'b0;
    doneA = 1'b0; doneB = 1'b0;

    case (state)

      //---------------------------------------------------------------------
      IDLE:
        if (conv_start) begin
          if (slrx_cmd == CONV_SETUP) begin
            next_state = READ_KERNEL;
          end
          else if (slrx_cmd == CONV_WINDOW) begin
            // init both blocks -- internal loop starts at row-pair 0; the FSM will loop
            // ALL row-pairs (0..half_rows-1) before asserting DONE (no per-row-pair
            // host round trip)
            out_row_cnt_ps = 0;
            colA_ps = 0;            colB_ps = 0;
            p_validA_ps = 1'b0;     p_validB_ps = 1'b0;
            wr_validA_ps = 1'b0;    wr_validB_ps = 1'b0;
            last_served_ps = 1'b1;  // so block A is served first
            baseA_ps  = conv_arr_out_addr + (out_row_cnt_ps * conv_arr_out_dim);
            baseB_ps  = conv_arr_out_addr + ((out_row_cnt_ps + half_rows) * conv_arr_out_dim);
            activeB_ps = ((out_row_cnt_ps + half_rows) < conv_arr_out_dim);
            // start loading block A's 5 input rows
            arr_in_row_addr_ps  = conv_arr_in_addr + (out_row_cnt_ps * conv_arr_in_dim);
            buf_load_row_idx_ps = 0;
            next_state = LOAD_A;
          end
        end

      //---------------------------------------------------------------------
      READ_KERNEL: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = conv_kernel_addr;
        mem_intf_read.mem_size_bytes = KERNEL_SIZE;
        if (mem_intf_read.mem_valid) begin
          kernel_ps  = mem_intf_read.mem_data[KERNEL_SIZE-1:0];
          next_state = DONE;
        end
      end

      //---------------------------------------------------------------------
      LOAD_A: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = arr_in_row_addr;
        mem_intf_read.mem_size_bytes = DIM_MAX_SIZE;
        arr_in_row_addr_ps           = arr_in_row_addr + conv_arr_in_dim;

        if (mem_intf_read.mem_valid) begin
          bufA_ps[buf_load_row_idx] = mem_intf_read.mem_data;
          if (is_last_load_row) begin
            mem_intf_read.mem_req = 0;
            buf_load_row_idx_ps   = 0;
            if (activeB) begin
              // switch to loading block B's region
              arr_in_row_addr_ps = conv_arr_in_addr + (row_b * conv_arr_in_dim);
              next_state         = LOAD_B;
            end
            else begin
              next_state = STREAM;
            end
          end
          else begin
            mem_intf_read.mem_start_addr = arr_in_row_addr;
            buf_load_row_idx_ps          = buf_load_row_idx + 1;
          end
        end
      end

      //---------------------------------------------------------------------
      LOAD_B: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = arr_in_row_addr;
        mem_intf_read.mem_size_bytes = DIM_MAX_SIZE;
        arr_in_row_addr_ps           = arr_in_row_addr + conv_arr_in_dim;

        if (mem_intf_read.mem_valid) begin
          bufB_ps[buf_load_row_idx] = mem_intf_read.mem_data;
          if (is_last_load_row) begin
            mem_intf_read.mem_req = 0;
            next_state            = STREAM;
          end
          else begin
            mem_intf_read.mem_start_addr = arr_in_row_addr;
            buf_load_row_idx_ps          = buf_load_row_idx + 1;
          end
        end
      end

      //---------------------------------------------------------------------
      // STREAM : both blocks compute in parallel; one shared write port.
      // Compute is now a 2-stage pipeline (stage1: fetch+partial-sum,
      // stage2: finalize+forward to the existing write-hold register); the
      // write port arbitration itself is UNCHANGED from the proven design.
      //---------------------------------------------------------------------
      STREAM: begin
        pendingA = wr_validA;
        pendingB = wr_validB && activeB;

        // round-robin arbiter for the single write port (unchanged)
        if (pendingA && pendingB) begin
          grantA = (last_served == 1'b1);   // B served last -> serve A now
          grantB = ~grantA;
        end
        else begin
          grantA = pendingA;
          grantB = pendingB;
        end

        // drive the write port from the granted block
        mem_intf_write.mem_req = pendingA || pendingB;
        if (grantA) begin
          mem_intf_write.mem_data       = wr_byteA;
          mem_intf_write.mem_start_addr = baseA + wr_colA;
        end
        else begin
          mem_intf_write.mem_data       = wr_byteB;
          mem_intf_write.mem_start_addr = baseB + wr_colB;
        end

        ackA = grantA && mem_intf_write.mem_ack;
        ackB = grantB && mem_intf_write.mem_ack;
        if (ackA) last_served_ps = 1'b0;
        if (ackB) last_served_ps = 1'b1;

        slot_freeA = (!wr_validA) || ackA;
        slot_freeB = (!wr_validB) || ackB;

        // ---- stage 2 -> write-hold : forward the finalized byte for the
        // column that finished pipeline stage 1 last cycle ----
        if (slot_freeA) begin
          if (p_validA) begin
            wr_byteA_ps  = resultA_final;
            wr_colA_ps   = p_colA;
            wr_validA_ps = 1'b1;
          end
          else begin
            wr_validA_ps = 1'b0;
          end
        end

        if (activeB && slot_freeB) begin
          if (p_validB) begin
            wr_byteB_ps  = resultB_final;
            wr_colB_ps   = p_colB;
            wr_validB_ps = 1'b1;
          end
          else begin
            wr_validB_ps = 1'b0;
          end
        end

        // ---- stage 1 -> stage 2 : fetch the next column's window and
        // compute its partial sums, as soon as the stage-2 slot has room ----
        slot_free_pA = (!p_validA) || slot_freeA;
        slot_free_pB = (!p_validB) || slot_freeB;

        if (slot_free_pA) begin
          if (colA < conv_arr_out_dim) begin
            p1A_ps      = partial1A;
            p2A_ps      = partial2A;
            p_colA_ps   = colA;
            p_validA_ps = 1'b1;
            colA_ps     = colA + 1;
          end
          else begin
            p_validA_ps = 1'b0;
          end
        end

        if (activeB && slot_free_pB) begin
          if (colB < conv_arr_out_dim) begin
            p1B_ps      = partial1B;
            p2B_ps      = partial2B;
            p_colB_ps   = colB;
            p_validB_ps = 1'b1;
            colB_ps     = colB + 1;
          end
          else begin
            p_validB_ps = 1'b0;
          end
        end

        // termination : both blocks' ENTIRE pipeline (stage1, stage2, and
        // the write-hold register) must be drained -- not just the column
        // counter -- since results can still be in flight for 2 more cycles
        // after the last column is fetched.
        doneA = (colA >= conv_arr_out_dim) && !p_validA && !wr_validA;
        doneB = (!activeB) || ((colB >= conv_arr_out_dim) && !p_validB && !wr_validB);
        if (doneA && doneB) begin
          if (out_row_cnt < (half_rows - 1)) begin
            // more row-pairs remain: loop straight back into LOAD_A with zero bubble
            // cycles -- no DONE/IDLE/host round trip between row-pairs
            out_row_cnt_ps = out_row_cnt + 1'b1;
            colA_ps = 0;            colB_ps = 0;
            p_validA_ps = 1'b0;     p_validB_ps = 1'b0;
            wr_validA_ps = 1'b0;    wr_validB_ps = 1'b0;
            last_served_ps = 1'b1;
            baseA_ps  = conv_arr_out_addr + (out_row_cnt_ps * conv_arr_out_dim);
            baseB_ps  = conv_arr_out_addr + ((out_row_cnt_ps + half_rows) * conv_arr_out_dim);
            activeB_ps = ((out_row_cnt_ps + half_rows) < conv_arr_out_dim);
            arr_in_row_addr_ps  = conv_arr_in_addr + (out_row_cnt_ps * conv_arr_in_dim);
            buf_load_row_idx_ps = 0;
            next_state = LOAD_A;
          end
          else begin
            next_state = DONE;
          end
        end
      end

      //---------------------------------------------------------------------
      DONE: begin
        conv_done = 1;
        if (clear_done_on_read)
          next_state = IDLE;
      end

    endcase
  end

  //===========================================================================
  // Sequential
  //===========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= IDLE;
      bufA <= 0; bufB <= 0; kernel <= 0;
      arr_in_row_addr  <= 0;
      buf_load_row_idx <= 0;
      colA <= 0; colB <= 0;
      baseA <= 0; baseB <= 0;
      p1A <= 0; p2A <= 0; p_colA <= 0; p_validA <= 1'b0;
      p1B <= 0; p2B <= 0; p_colB <= 0; p_validB <= 1'b0;
      wr_validA <= 1'b0; wr_byteA <= 0; wr_colA <= 0;
      wr_validB <= 1'b0; wr_byteB <= 0; wr_colB <= 0;
      activeB <= 1'b0; last_served <= 1'b1;
      out_row_cnt <= 0;
    end
    else begin
      state            <= next_state;
      bufA <= bufA_ps; bufB <= bufB_ps; kernel <= kernel_ps;
      arr_in_row_addr  <= arr_in_row_addr_ps;
      buf_load_row_idx <= buf_load_row_idx_ps;
      colA <= colA_ps; colB <= colB_ps;
      baseA <= baseA_ps; baseB <= baseB_ps;
      p1A <= p1A_ps; p2A <= p2A_ps; p_colA <= p_colA_ps; p_validA <= p_validA_ps;
      p1B <= p1B_ps; p2B <= p2B_ps; p_colB <= p_colB_ps; p_validB <= p_validB_ps;
      wr_validA <= wr_validA_ps; wr_byteA <= wr_byteA_ps; wr_colA <= wr_colA_ps;
      wr_validB <= wr_validB_ps; wr_byteB <= wr_byteB_ps; wr_colB <= wr_colB_ps;
      activeB <= activeB_ps; last_served <= last_served_ps;
      out_row_cnt <= out_row_cnt_ps;
    end
  end

  //===========================================================================
  // Calculation Functions -- SAME arithmetic as the proven single-cycle
  // version, just split across the term range so it can be summed over 2
  // cycles: calc_conv_partial_sum sums taps [term_start, term_end) only
  // (no bias, no ReLU/descale/saturate yet); calc_conv_finalize combines
  // both partial sums + bias then applies ReLU/descale/saturate exactly as
  // calc_conv_win used to in one shot.
  //===========================================================================
  function automatic logic signed [MAX_DOT_PROD_WIDTH-1:0] calc_conv_partial_sum;
      input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel;
      input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win;
      input int term_start;
      input int term_end; // exclusive

      logic signed [MAX_DOT_PROD_WIDTH-1:0] acc;
      logic signed [MAX_DOT_PROD_WIDTH-1:0] mult;
      begin
        acc = 0;
        for (int t = term_start; t < term_end; t++) begin
          mult = $signed({1'b0, conv_win[t / KERNEL_DIM][t % KERNEL_DIM]}) *
                 $signed(kernel[t / KERNEL_DIM][t % KERNEL_DIM]);
          acc = acc + mult;
        end
        calc_conv_partial_sum = acc;
      end
  endfunction

  function automatic logic [7:0] calc_conv_finalize;
      input signed [MAX_DOT_PROD_WIDTH-1:0] partial1;
      input signed [MAX_DOT_PROD_WIDTH-1:0] partial2;
      input signed [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;

      logic signed [MAX_DOT_PROD_WIDTH-1:0] acc;
      logic signed [MAX_DOT_PROD_WIDTH-1:0] descale_val;
      begin
        acc = conv_bias_val + partial1 + partial2;
        if (acc < 0) begin
          calc_conv_finalize = 8'd0;
        end
        else begin
          descale_val = acc >>> 8;
          if (descale_val > 255)
            calc_conv_finalize = 8'd255;
          else
            calc_conv_finalize = descale_val[7:0];
        end
      end
  endfunction

endmodule
