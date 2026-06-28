import xbox_def_pkg::*;
import slrx_def_pkg::*;

module conv (
  input   clk,
  input   rst_n,
  slrx_regs_intrf.xlr slrx_regs_intrf,
  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  localparam DIM_MAX_SIZE       = 32;
  localparam KERNEL_DIM         = 5;
  localparam KERNEL_SIZE        = KERNEL_DIM*KERNEL_DIM;
  localparam MAX_DOT_PROD_WIDTH = 16+$clog2(KERNEL_SIZE);
  localparam ARR_IDX_W          = $clog2(DIM_MAX_SIZE);

  enum { IDLE, READ_KERNEL, READ_ROWS, WINDOW, CALC, WRITE, DONE } next_state, state;

  logic conv_start, conv_done, clear_done_on_read;
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel, kernel_ps;
  logic [KERNEL_DIM-1:0][DIM_MAX_SIZE-1:0][7:0] conv_rows_buf, conv_rows_buf_ps;
  logic [XMEM_ADDR_WIDTH-1:0] conv_kernel_addr, conv_arr_in_addr, conv_arr_out_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_rslt_out_addr, conv_rslt_out_addr_ps;
  logic [XMEM_ADDR_WIDTH-1:0] arr_in_row_addr, arr_in_row_addr_ps;
  logic [MAX_DOT_PROD_WIDTH-1:0] conv_bias_val;
  logic [ARR_IDX_W:0] conv_arr_in_dim, conv_arr_out_dim;
  logic [ARR_IDX_W-1:0] conv_out_row_idx, conv_out_col_idx;
  logic [ARR_IDX_W-1:0] buf_load_row_idx, buf_load_row_idx_ps;
  logic is_last_load_row, conv_active;
  logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win, conv_win_ps;
  logic [7:0] conv_out_val, conv_out_val_ps;
  slrx_cmd_t slrx_cmd;

  assign slrx_regs_intrf.xlr_done = conv_done;
  assign slrx_cmd       = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);
  assign conv_active    = (slrx_cmd == CONV_SETUP) || (slrx_cmd == CONV_WINDOW);
  assign conv_start     = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && conv_active;
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
  assign conv_rslt_out_addr_ps = conv_arr_out_addr + (conv_out_row_idx * conv_arr_out_dim) + conv_out_col_idx;

  always_comb begin
    next_state = state;
    mem_intf_read.mem_size_bytes  = 0;
    mem_intf_read.mem_start_addr  = 0;
    mem_intf_read.mem_req         = 0;
    mem_intf_write.mem_size_bytes = 1;
    mem_intf_write.mem_data       = conv_out_val;
    mem_intf_write.mem_start_addr = conv_rslt_out_addr;
    mem_intf_write.mem_req        = 0;
    conv_done                     = 0;
    buf_load_row_idx_ps           = buf_load_row_idx;
    conv_rows_buf_ps              = conv_rows_buf;
    arr_in_row_addr_ps            = arr_in_row_addr;
    kernel_ps                     = kernel;

    case (state)
      IDLE: if (conv_start) begin
        if (slrx_cmd == CONV_SETUP) next_state = READ_KERNEL;
        else if (slrx_cmd == CONV_WINDOW) begin
          next_state = READ_ROWS;
          arr_in_row_addr_ps = conv_arr_in_addr + (conv_out_row_idx * conv_arr_in_dim);
        end
        buf_load_row_idx_ps = 0;
      end
      READ_KERNEL: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = conv_kernel_addr;
        mem_intf_read.mem_size_bytes = KERNEL_SIZE;
        if (mem_intf_read.mem_valid) begin
          for (int ki = 0; ki < KERNEL_DIM; ki++)
            for (int kj = 0; kj < KERNEL_DIM; kj++)
              kernel_ps[ki][kj] = mem_intf_read.mem_data[(ki*KERNEL_DIM)+kj];
          next_state = DONE;
        end
      end
      READ_ROWS: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = arr_in_row_addr;
        mem_intf_read.mem_size_bytes = DIM_MAX_SIZE;
        arr_in_row_addr_ps           = arr_in_row_addr + conv_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          conv_rows_buf_ps[buf_load_row_idx] = mem_intf_read.mem_data;
          if (is_last_load_row) begin
            next_state            = WINDOW;
            mem_intf_read.mem_req = 0;
          end else begin
            mem_intf_read.mem_start_addr = arr_in_row_addr;
            buf_load_row_idx_ps          = buf_load_row_idx + 1;
          end
        end
      end
      WINDOW: next_state = CALC;
      CALC:   next_state = WRITE;
      WRITE: begin
        mem_intf_write.mem_req = 1;
        if (mem_intf_write.mem_ack) begin
          next_state             = DONE;
          mem_intf_write.mem_req = 0;
        end
      end
      DONE: begin
        conv_done = 1;
        if (clear_done_on_read) next_state = IDLE;
      end
    endcase
  end

  always_comb begin
    conv_win_ps = conv_win;
    if (state == WINDOW) begin
      for (int i = 0; i < KERNEL_DIM; i++)
        for (int j = 0; j < KERNEL_DIM; j++)
          conv_win_ps[i][j] = conv_rows_buf[i][conv_out_col_idx + j];
    end
  end

  assign conv_out_val_ps = calc_conv_win(kernel, conv_bias_val, conv_win);

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
    end else begin
      state              <= next_state;
      arr_in_row_addr    <= arr_in_row_addr_ps;
      buf_load_row_idx   <= buf_load_row_idx_ps;
      kernel             <= kernel_ps;
      conv_rows_buf      <= conv_rows_buf_ps;
      conv_out_val       <= conv_out_val_ps;
      conv_win           <= conv_win_ps;
      conv_rslt_out_addr <= conv_rslt_out_addr_ps;
    end
  end

  function automatic logic [7:0] calc_conv_win;
    input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] kernel;
    input signed [MAX_DOT_PROD_WIDTH-1:0]        conv_bias_val;
    input [KERNEL_DIM-1:0][KERNEL_DIM-1:0][7:0] conv_win;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] acc;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] shifted;
    logic [7:0] result;
    begin
      acc = conv_bias_val;
      for (int i = 0; i < KERNEL_DIM; i++)
        for (int j = 0; j < KERNEL_DIM; j++)
          begin
            logic signed [8:0]  k_val;
            logic signed [8:0]  w_val;
            logic signed [17:0] prod;
            k_val = $signed(kernel[i][j]);
            w_val = $signed({1'b0, conv_win[i][j]});
            prod  = k_val * w_val;
            acc   = acc + MAX_DOT_PROD_WIDTH'(signed'(prod));
          end
      shifted = acc >>> 7;
      if (shifted[MAX_DOT_PROD_WIDTH-1]) result = 8'd0;
      else if (shifted > 255) result = 8'd255;
      else result = shifted[7:0];
      calc_conv_win = result;
    end
  endfunction

endmodule
