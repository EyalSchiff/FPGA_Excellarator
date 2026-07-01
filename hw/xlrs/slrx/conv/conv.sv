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
//   Each block has its own 5-row input buffer and its own 25-MAC datapath,
//   and streams its output row left->right at 1 pixel/cycle. The single write
//   port is shared by a round-robin arbiter (one byte written per cycle).
//
//   Arithmetic (bias + sum(pixel*weight), ReLU, descale >>>8, saturate) is
//   IDENTICAL to the proven single-block version.
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
  logic [ARR_IDX_W:0]         colA, colA_ps;     // compute column (block A)
  logic [ARR_IDX_W:0]         colB, colB_ps;     // compute column (block B)
  logic [XMEM_ADDR_WIDTH-1:0] baseA, baseA_ps;   // output base addr (row A)
  logic [XMEM_ADDR_WIDTH-1:0] baseB, baseB_ps;   // output base addr (row B)

  logic                wr_validA, wr_validA_ps;  // output pipeline reg A
  logic [7:0]          wr_byteA,  wr_byteA_ps;
  logic [ARR_IDX_W:0]  wr_colA,   wr_colA_ps;

  logic                wr_validB, wr_validB_ps;  // output pipeline reg B
  logic [7:0]          wr_byteB,  wr_byteB_ps;
  logic [ARR_IDX_W:0]  wr_colB,   wr_colB_ps;

  logic                activeB, activeB_ps;      // block B has a valid row this call
  logic                last_served, last_served_ps; // 0=A served last, 1=B served last

  //===========================================================================
  // Two parallel 25-MAC blocks (combinational window + dot product)
  //===========================================================================
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] winA, winB;
  logic [7:0]                                 resultA, resultB;

  always_comb begin
    for (int i = 0; i < KERNEL_DIM; i++)
      for (int j = 0; j < KERNEL_DIM; j++)
        winA[i][j] = bufA[i][colA + j];
    resultA = calc_conv_win(kernel, conv_bias_val, winA);

    for (int i = 0; i < KERNEL_DIM; i++)
      for (int j = 0; j < KERNEL_DIM; j++)
        winB[i][j] = bufB[i][colB + j];
    resultB = calc_conv_win(kernel, conv_bias_val, winB);
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
    wr_validA_ps = wr_validA; wr_byteA_ps = wr_byteA; wr_colA_ps = wr_colA;
    wr_validB_ps = wr_validB; wr_byteB_ps = wr_byteB; wr_colB_ps = wr_colB;
    activeB_ps = activeB; last_served_ps = last_served;
    out_row_cnt_ps = out_row_cnt;

    pendingA = 1'b0; pendingB = 1'b0; grantA = 1'b0; grantB = 1'b0;
    ackA = 1'b0; ackB = 1'b0; slot_freeA = 1'b0; slot_freeB = 1'b0;
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
      // STREAM : both blocks compute in parallel; one shared write port
      //---------------------------------------------------------------------
      STREAM: begin
        pendingA = wr_validA;
        pendingB = wr_validB && activeB;

        // round-robin arbiter for the single write port
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

        // compute stage : block A
        if (slot_freeA) begin
          if (colA < conv_arr_out_dim) begin
            wr_byteA_ps = resultA;
            wr_colA_ps  = colA;
            wr_validA_ps= 1'b1;
            colA_ps     = colA + 1;
          end
          else begin
            wr_validA_ps = 1'b0;
          end
        end

        // compute stage : block B
        if (activeB && slot_freeB) begin
          if (colB < conv_arr_out_dim) begin
            wr_byteB_ps = resultB;
            wr_colB_ps  = colB;
            wr_validB_ps= 1'b1;
            colB_ps     = colB + 1;
          end
          else begin
            wr_validB_ps = 1'b0;
          end
        end

        // termination : both rows fully computed and drained
        doneA = (colA >= conv_arr_out_dim) && !wr_validA;
        doneB = (!activeB) || ((colB >= conv_arr_out_dim) && !wr_validB);
        if (doneA && doneB) begin
          if (out_row_cnt < (half_rows - 1)) begin
            // more row-pairs remain: loop straight back into LOAD_A with zero bubble
            // cycles -- no DONE/IDLE/host round trip between row-pairs
            out_row_cnt_ps = out_row_cnt + 1'b1;
            colA_ps = 0;            colB_ps = 0;
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
      wr_validA <= wr_validA_ps; wr_byteA <= wr_byteA_ps; wr_colA <= wr_colA_ps;
      wr_validB <= wr_validB_ps; wr_byteB <= wr_byteB_ps; wr_colB <= wr_colB_ps;
      activeB <= activeB_ps; last_served <= last_served_ps;
      out_row_cnt <= out_row_cnt_ps;
    end
  end

  //===========================================================================
  // Calculation Function (UNCHANGED): bias + sum(pixel*weight), ReLU, >>>8, sat
  //===========================================================================
  function automatic logic [7:0] calc_conv_win;
      input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel;
      input signed [MAX_DOT_PROD_WIDTH-1:0]       conv_bias_val;
      input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win;

      logic signed [MAX_DOT_PROD_WIDTH-1:0] acc;
      logic signed [MAX_DOT_PROD_WIDTH-1:0] mult;
      logic signed [MAX_DOT_PROD_WIDTH-1:0] descale_val;
      begin
        acc = conv_bias_val;
        for (int kernel_row_idx = 0; kernel_row_idx < KERNEL_DIM; kernel_row_idx++) begin
          for (int kernel_col_idx = 0; kernel_col_idx < KERNEL_DIM; kernel_col_idx++) begin
            mult = $signed({1'b0, conv_win[kernel_row_idx][kernel_col_idx]}) *
                   $signed(kernel[kernel_row_idx][kernel_col_idx]);
            acc  = acc + mult;
          end
        end
        if (acc < 0) begin
          calc_conv_win = 8'd0;
        end
        else begin
          descale_val = acc >>> 8;
          if (descale_val > 255)
            calc_conv_win = 8'd255;
          else
            calc_conv_win = descale_val[7:0];
        end
      end
  endfunction

endmodule