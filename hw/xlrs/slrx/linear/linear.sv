import xbox_def_pkg::*;
import slrx_def_pkg::*;

module linear (
  input   clk,
  input   rst_n,
  slrx_regs_intrf.xlr slrx_regs_intrf,
  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  // Contract: ONE LIN_CALC computes TWO output elements.
  //   Block A : out_col = lin_out_col_idx
  //   Block B : out_col = lin_out_col_idx + half_cols
  //   half_cols = ceil(out_dim/2). C driver loops c = 0..half_cols-1.
  // in_vec is loaded once at LIN_SETUP and shared (read-only) by both blocks.

  enum {IDLE, READ_IN_VEC, READ_WGTA, READ_BIASA, READ_WGTB, READ_BIASB, CALC, WRITE, DONE} next_state, state;

  localparam DIM_MAX_SIZE = 32;
  localparam MAX_DOT_PROD_WIDTH = 16+$clog2(DIM_MAX_SIZE);
  localparam ARR_IDX_W = $clog2(DIM_MAX_SIZE);

  logic lin_start, lin_done, clear_done_on_read, lin_active;

  logic [DIM_MAX_SIZE-1:0][7:0] in_vec, in_vec_ps; // shared input, loaded once

  logic [DIM_MAX_SIZE-1:0][7:0] wgtA, wgtA_ps;
  logic [DIM_MAX_SIZE-1:0][7:0] wgtB, wgtB_ps;
  logic signed [31:0] biasA, biasA_ps;
  logic signed [31:0] biasB, biasB_ps;

  logic [XMEM_ADDR_WIDTH-1:0] lin_wgt_arr_addr, lin_arr_in_addr, lin_arr_out_addr, lin_bias_vec_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_rslt_out_addr, lin_rslt_out_addr_ps;

  logic [ARR_IDX_W:0] lin_arr_in_dim, lin_arr_out_dim;
  logic [ARR_IDX_W-1:0] lin_out_col_idx;
  logic [ARR_IDX_W:0] half_cols, col_b;
  logic activeB, activeB_ps;

  logic [7:0] lin_out_valA, lin_out_valA_ps;
  logic [7:0] lin_out_valB, lin_out_valB_ps;

  assign slrx_regs_intrf.xlr_done = lin_done;

  slrx_cmd_t slrx_cmd;
  assign slrx_cmd            = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);
  assign lin_active          = (slrx_cmd==LIN_SETUP) || (slrx_cmd==LIN_CALC);
  assign lin_start           = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && lin_active;
  assign clear_done_on_read  = lin_active && slrx_regs_intrf.xlr_done_ack;

  assign lin_wgt_arr_addr    = slrx_regs_intrf.host_regs[WGT_ADDR_RI];
  assign lin_bias_vec_addr   = slrx_regs_intrf.host_regs[LIN_BIAS_ADDR_RI];
  assign lin_arr_in_addr     = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];
  assign lin_arr_out_addr    = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI];
  assign lin_arr_in_dim      = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];
  assign lin_arr_out_dim     = slrx_regs_intrf.host_regs[ARR_OUT_DIM_RI];
  assign lin_out_col_idx     = slrx_regs_intrf.host_regs[OUT_COL_IDX_RI];

  assign half_cols = (lin_arr_out_dim + 1) >> 1;        // ceil(out_dim/2) -- not used for stepping, A/B are this call's pair
  assign col_b     = lin_out_col_idx + 1;                // B is simply "the next column" within this pair
  // NOTE: driver calls with out_col_idx = 0,2,4,... ; block B handles col+1

  function automatic logic [7:0] calc_lin_element(
      input [DIM_MAX_SIZE-1:0][7:0] wv,
      input signed [MAX_DOT_PROD_WIDTH-1:0] bv,
      input [DIM_MAX_SIZE-1:0][7:0] iv
  );
    logic signed [MAX_DOT_PROD_WIDTH-1:0] accum;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] descale;
    logic signed [16:0] prod;
    begin
      accum = bv;
      for (int i = 0; i < DIM_MAX_SIZE; i++) begin
        prod  = $signed(wv[i]) * $signed({1'b0, iv[i]});
        accum = accum + prod;
      end
      descale = accum >>> 8;
      if (descale <= 0) calc_lin_element = 8'd0;
      else if (descale > 255) calc_lin_element = 8'd255;
      else calc_lin_element = descale[7:0];
    end
  endfunction

  always_comb begin
    next_state = state;

    mem_intf_read.mem_size_bytes  = 0;
    mem_intf_read.mem_start_addr  = 0;
    mem_intf_read.mem_req         = 0;

    mem_intf_write.mem_size_bytes = activeB ? 2 : 1;
    mem_intf_write.mem_data       = {lin_out_valB, lin_out_valA};
    mem_intf_write.mem_start_addr = lin_rslt_out_addr;
    mem_intf_write.mem_req        = 0;

    lin_done = 0;

    in_vec_ps = in_vec;
    wgtA_ps = wgtA; wgtB_ps = wgtB;
    biasA_ps = biasA; biasB_ps = biasB;
    lin_out_valA_ps = lin_out_valA; lin_out_valB_ps = lin_out_valB;
    lin_rslt_out_addr_ps = lin_arr_out_addr + lin_out_col_idx;
    activeB_ps = activeB;

    case (state)

      IDLE: if (lin_start) begin
        if (slrx_cmd == LIN_SETUP) begin
          next_state = READ_IN_VEC;
        end
        else if (slrx_cmd == LIN_CALC) begin
          activeB_ps = (col_b < lin_arr_out_dim);
          next_state = READ_WGTA;
        end
      end

      READ_IN_VEC: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_arr_in_addr;
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            in_vec_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          next_state = DONE;
        end
      end

      READ_WGTA: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_wgt_arr_addr + (lin_out_col_idx * lin_arr_in_dim);
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            wgtA_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          next_state = READ_BIASA;
        end
      end

      READ_BIASA: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_bias_vec_addr + (4*lin_out_col_idx);
        mem_intf_read.mem_size_bytes = 4;
        if (mem_intf_read.mem_valid) begin
          biasA_ps = mem_intf_read.mem_data[3:0];
          mem_intf_read.mem_req = 0;
          next_state = activeB ? READ_WGTB : CALC;
        end
      end

      READ_WGTB: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_wgt_arr_addr + (col_b * lin_arr_in_dim);
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            wgtB_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          next_state = READ_BIASB;
        end
      end

      READ_BIASB: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_bias_vec_addr + (4*col_b);
        mem_intf_read.mem_size_bytes = 4;
        if (mem_intf_read.mem_valid) begin
          biasB_ps = mem_intf_read.mem_data[3:0];
          mem_intf_read.mem_req = 0;
          next_state = CALC;
        end
      end

      CALC: begin
        lin_out_valA_ps = calc_lin_element(wgtA, biasA[MAX_DOT_PROD_WIDTH-1:0], in_vec);
        if (activeB)
          lin_out_valB_ps = calc_lin_element(wgtB, biasB[MAX_DOT_PROD_WIDTH-1:0], in_vec);
        next_state = WRITE;
      end

      WRITE: begin
        mem_intf_write.mem_req = 1;
        if (mem_intf_write.mem_ack) begin
          next_state = DONE;
          mem_intf_write.mem_req = 0;
        end
      end

      DONE: begin
        lin_done = 1;
        if (clear_done_on_read) next_state = IDLE;
      end

    endcase
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= IDLE;
      in_vec        <= 0;
      wgtA <= 0; wgtB <= 0;
      biasA <= 0; biasB <= 0;
      lin_out_valA  <= 0; lin_out_valB <= 0;
      lin_rslt_out_addr <= 0;
      activeB       <= 1'b0;
    end else begin
      state         <= next_state;
      in_vec        <= in_vec_ps;
      wgtA <= wgtA_ps; wgtB <= wgtB_ps;
      biasA <= biasA_ps; biasB <= biasB_ps;
      lin_out_valA  <= lin_out_valA_ps; lin_out_valB <= lin_out_valB_ps;
      lin_rslt_out_addr <= lin_rslt_out_addr_ps;
      activeB       <= activeB_ps;
    end
  end

endmodule
