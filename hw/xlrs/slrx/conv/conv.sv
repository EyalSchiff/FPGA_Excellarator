//=============================================================================
// File: conv.sv
// Description: Convolution Accelerator (5x5) - OPTIMIZED
//   Same proven datapath / arithmetic as the working version.
//   Optimization: the 5 input rows are CACHED across CONV_WINDOW calls.
//   Because the C driver sweeps out_col_idx with out_row_idx held constant,
//   consecutive windows in the same output row reuse the buffered rows and
//   skip READ_ROWS entirely. Only the first pixel of each output row reloads.
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
     READ_ROWS,
     WINDOW,
     CALC,
     WRITE,
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
  // Control Signals
  //===========================================================================
  logic conv_start;
  logic conv_done;
  logic clear_done_on_read;

  //===========================================================================
  // Kernel Storage
  //===========================================================================
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel;
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel_ps;

  //===========================================================================
  // Input Data Buffer (5 rows x up to 32 bytes)
  //===========================================================================
  logic [KERNEL_DIM-1:0][DIM_MAX_SIZE-1:0][7:0] conv_rows_buf;
  logic [KERNEL_DIM-1:0][DIM_MAX_SIZE-1:0][7:0] conv_rows_buf_ps;

  //===========================================================================
  // Memory Addresses
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] conv_kernel_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_in_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_out_addr;

  logic [XMEM_ADDR_WIDTH-1:0] conv_rslt_out_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_rslt_out_addr_ps;

  //===========================================================================
  // Layer Configuration
  //===========================================================================
  logic [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;

  logic [ARR_IDX_W:0] conv_arr_in_dim;
  logic [ARR_IDX_W:0] conv_arr_out_dim;

  logic [ARR_IDX_W-1:0] conv_out_row_idx;
  logic [ARR_IDX_W-1:0] conv_out_col_idx;

  //===========================================================================
  // Memory Addressing
  //===========================================================================
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr;
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr_ps;

  //===========================================================================
  // Output Value
  //===========================================================================
  logic [7:0] conv_out_val;
  logic [7:0] conv_out_val_ps;

  //===========================================================================
  // Buffer Control
  //===========================================================================
  logic [ARR_IDX_W-1:0] buf_load_row_idx;
  logic [ARR_IDX_W-1:0] buf_load_row_idx_ps;
  logic is_last_load_row;

  //===========================================================================
  // ROW CACHE  (the optimization)
  //   cached_row_idx : which out_row_idx the buffer currently holds
  //   rows_valid     : buffer contents are valid for that row
  //===========================================================================
  logic [ARR_IDX_W-1:0] cached_row_idx;
  logic [ARR_IDX_W-1:0] cached_row_idx_ps;
  logic                 rows_valid;
  logic                 rows_valid_ps;

  logic conv_active;

  //===========================================================================
  // Internal column loop counter (HW looping optimization)
  //===========================================================================
  logic [ARR_IDX_W-1:0] loop_col_idx;
  logic [ARR_IDX_W-1:0] loop_col_idx_ps;
  logic is_last_col;

  //===========================================================================
  // Host Register Interface
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
  assign conv_out_row_idx  = slrx_regs_intrf.host_regs[OUT_ROW_IDX_RI];
  assign conv_out_col_idx  = slrx_regs_intrf.host_regs[OUT_COL_IDX_RI];

  assign conv_arr_out_dim  = conv_arr_in_dim - (KERNEL_DIM - 1);

  assign is_last_load_row  = (buf_load_row_idx == (KERNEL_DIM - 1));

  assign is_last_col = (loop_col_idx == (conv_arr_out_dim - 1));

  assign conv_rslt_out_addr_ps = conv_arr_out_addr +
                                 (conv_out_row_idx * conv_arr_out_dim) +
                                 loop_col_idx;

  //===========================================================================
  // FSM - Combinational
  //===========================================================================
  always_comb begin

    next_state = state;

    mem_intf_read.mem_size_bytes  = 0;
    mem_intf_read.mem_start_addr  = 0;
    mem_intf_read.mem_req         = 0;

    mem_intf_write.mem_size_bytes = 1;
    mem_intf_write.mem_data       = conv_out_val;
    mem_intf_write.mem_start_addr = conv_rslt_out_addr;
    mem_intf_write.mem_req        = 0;

    conv_done = 0;

    buf_load_row_idx_ps = buf_load_row_idx;
    conv_rows_buf_ps    = conv_rows_buf;
    arr_in_row_addr_ps  = arr_in_row_addr;
    kernel_ps           = kernel;
    cached_row_idx_ps   = cached_row_idx;
    rows_valid_ps       = rows_valid;
    loop_col_idx_ps     = loop_col_idx;

    case (state)

      //---------------------------------------------------------------------
      IDLE:
        if (conv_start) begin
          if (slrx_cmd == CONV_SETUP) begin
            next_state    = READ_KERNEL;
            rows_valid_ps = 1'b0;   // new layer/kernel -> invalidate cached rows
          end
          else if (slrx_cmd == CONV_WINDOW) begin
            loop_col_idx_ps = conv_out_col_idx; // SW always sends 0
            if (rows_valid && (conv_out_row_idx == cached_row_idx)) begin
              // CACHE HIT: the 5 needed rows are already buffered -> skip reads
              next_state = WINDOW;
            end
            else begin
              // CACHE MISS: load the 5 rows for this output row
              next_state          = READ_ROWS;
              arr_in_row_addr_ps  = conv_arr_in_addr + (conv_out_row_idx * conv_arr_in_dim);
              buf_load_row_idx_ps = 0;
            end
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
      READ_ROWS: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = arr_in_row_addr;
        mem_intf_read.mem_size_bytes = DIM_MAX_SIZE;

        arr_in_row_addr_ps = arr_in_row_addr + conv_arr_in_dim;

        if (mem_intf_read.mem_valid) begin
          conv_rows_buf_ps[buf_load_row_idx] = mem_intf_read.mem_data;

          if (is_last_load_row) begin
            next_state            = WINDOW;
            mem_intf_read.mem_req = 0;
            cached_row_idx_ps     = conv_out_row_idx;  // remember what is buffered
            rows_valid_ps         = 1'b1;
          end
          else begin
            mem_intf_read.mem_start_addr = arr_in_row_addr;
            buf_load_row_idx_ps          = buf_load_row_idx + 1;
          end
        end
      end

      //---------------------------------------------------------------------
      WINDOW:
        next_state = CALC;

      //---------------------------------------------------------------------
      CALC:
        next_state = WRITE;

      //---------------------------------------------------------------------
      WRITE: begin
        mem_intf_write.mem_req = 1;
        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;
          if (is_last_col) begin
            next_state = DONE;
          end else begin
            loop_col_idx_ps = loop_col_idx + 1;
            next_state      = WINDOW; // compute next column, rows already cached
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
  // Window Extraction
  //===========================================================================
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win_ps;
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win;

  always_comb begin
    conv_win_ps = conv_win;
    if (state == WINDOW) begin
      for (int i = 0; i < KERNEL_DIM; i++)
        for (int j = 0; j < KERNEL_DIM; j++)
          conv_win_ps[i][j] = conv_rows_buf[i][loop_col_idx + j];
    end
  end

  //===========================================================================
  // Convolution Calculation (identical arithmetic to the working version)
  //===========================================================================
  assign conv_out_val_ps = calc_conv_win(kernel, conv_bias_val, conv_win);

  //===========================================================================
  // Sequential
  //===========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state              <= IDLE;
      arr_in_row_addr    <= 0;
      buf_load_row_idx   <= 0;
      kernel             <= 0;
      conv_rows_buf      <= 0;
      conv_out_val       <= 0;
      conv_win           <= 0;
      conv_rslt_out_addr <= 0;
      cached_row_idx     <= 0;
      rows_valid         <= 1'b0;
      loop_col_idx       <= 0;
    end
    else begin
      state              <= next_state;
      arr_in_row_addr    <= arr_in_row_addr_ps;
      buf_load_row_idx   <= buf_load_row_idx_ps;
      kernel             <= kernel_ps;
      conv_rows_buf      <= conv_rows_buf_ps;
      conv_out_val       <= conv_out_val_ps;
      conv_win           <= conv_win_ps;
      conv_rslt_out_addr <= conv_rslt_out_addr_ps;
      cached_row_idx     <= cached_row_idx_ps;
      rows_valid         <= rows_valid_ps;
      loop_col_idx       <= loop_col_idx_ps;
    end
  end

  //===========================================================================
  // Calculation Function : bias + sum(pixel*weight), ReLU, descale >>> 8, saturate
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