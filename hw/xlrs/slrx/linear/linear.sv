import xbox_def_pkg::*;
import slrx_def_pkg::*;

module linear (
  input   clk,
  input   rst_n,
  slrx_regs_intrf.xlr slrx_regs_intrf,
  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  // Contract: ONE LIN_CALC command computes the ENTIRE output vector.
  //   The FSM internally loops column-pair index 0,2,4,... :
  //     Block A : out_col = out_col_cnt
  //     Block B : out_col = out_col_cnt + 1  (when in range)
  //   in_vec is loaded once at LIN_SETUP and shared (read-only) by both blocks.
  //   The host issues LIN_SETUP once and LIN_CALC once per layer (no per-pair
  //   host round trip) -- the FSM loops WRITE -> READ_WGTA directly between
  //   column-pairs with zero bubble cycles. BIASA/BIASB (adjacent int32_t
  //   array elements) are fetched in a single merged read.

  enum {IDLE, READ_IN_VEC, READ_WGTA, READ_WGTB, READ_BIAS, CALC, WRITE, DONE} next_state, state;

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
  logic [ARR_IDX_W:0] col_b;
  logic activeB, activeB_ps;

  // Internal output-column-pair loop counter: ONE LIN_CALC command now processes ALL
  // output columns (0,2,4,... ) inside the FSM, looping WRITE -> READ_WGTA directly
  // (zero bubble cycles) instead of returning to the host for a per-pair command.
  // OUT_COL_IDX_RI is no longer read by this module.
  logic [ARR_IDX_W:0] out_col_cnt, out_col_cnt_ps;

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

  assign col_b     = out_col_cnt + 1;                // B is simply "the next column" within this pair

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
    lin_rslt_out_addr_ps = lin_arr_out_addr + out_col_cnt;
    activeB_ps = activeB;
    out_col_cnt_ps = out_col_cnt;

    case (state)

      IDLE: if (lin_start) begin
        if (slrx_cmd == LIN_SETUP) begin
          next_state = READ_IN_VEC;
        end
        else if (slrx_cmd == LIN_CALC) begin
          // internal loop starts at column-pair 0; the FSM will loop ALL column-pairs
          // before asserting DONE (no per-pair host round trip)
          out_col_cnt_ps = 0;
          lin_rslt_out_addr_ps = lin_arr_out_addr;
          activeB_ps = (1 < lin_arr_out_dim);   // col_b for column 0 is column 1
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
        mem_intf_read.mem_start_addr = lin_wgt_arr_addr + (out_col_cnt * lin_arr_in_dim);
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          for (int i = 0; i < DIM_MAX_SIZE; i++)
            wgtA_ps[i] = (i < lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          next_state = activeB ? READ_WGTB : READ_BIAS;
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
          next_state = READ_BIAS;
        end
      end

      // biasA (out_col_cnt) and biasB (out_col_cnt+1) are adjacent int32_t elements of
      // the SAME bias array -- fetch both in one merged read (8 bytes, well within the
      // 32-byte bus width) instead of two separate round trips.
      READ_BIAS: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_bias_vec_addr + (4*out_col_cnt);
        mem_intf_read.mem_size_bytes = activeB ? 8 : 4;
        if (mem_intf_read.mem_valid) begin
          biasA_ps = mem_intf_read.mem_data[3:0];
          if (activeB) biasB_ps = mem_intf_read.mem_data[7:4];
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
          mem_intf_write.mem_req = 0;
          if ((out_col_cnt + 2) < lin_arr_out_dim) begin
            // more column-pairs remain: loop straight back into READ_WGTA with zero
            // bubble cycles -- no DONE/IDLE/host round trip between column-pairs
            out_col_cnt_ps        = out_col_cnt + 2;
            lin_rslt_out_addr_ps  = lin_arr_out_addr + out_col_cnt_ps;
            activeB_ps            = ((out_col_cnt_ps + 1) < lin_arr_out_dim);
            next_state            = READ_WGTA;
          end
          else begin
            next_state = DONE;
          end
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
      out_col_cnt   <= 0;
    end else begin
      state         <= next_state;
      in_vec        <= in_vec_ps;
      wgtA <= wgtA_ps; wgtB <= wgtB_ps;
      biasA <= biasA_ps; biasB <= biasB_ps;
      lin_out_valA  <= lin_out_valA_ps; lin_out_valB <= lin_out_valB_ps;
      lin_rslt_out_addr <= lin_rslt_out_addr_ps;
      activeB       <= activeB_ps;
      out_col_cnt   <= out_col_cnt_ps;
    end
  end

endmodule
