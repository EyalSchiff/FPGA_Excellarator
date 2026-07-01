#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h"

//---------------------------------------------------------------------------------------------------------------------------------
// Scalar reference (non-accelerated) : computes ONE output pixel.
//---------------------------------------------------------------------------------------------------------------------------------
void conv_window_nox(uint8_t* conv_arr_out,
                     uint8_t* conv_arr_in,
                     int      arr_in_dim,
                     int      out_row_idx,
                     int      out_col_idx,
                     int8_t*  kernel_w,
                     int32_t  kernel_b) {

    int out_dim = arr_in_dim - CONV_KERNEL_DIM + 1;

    int32_t acc = kernel_b;

    for (int kernel_row_idx = 0; kernel_row_idx < CONV_KERNEL_DIM; kernel_row_idx++) {
        for (int kernel_col_idx = 0; kernel_col_idx < CONV_KERNEL_DIM; kernel_col_idx++) {

            int in_row_idx = out_row_idx + kernel_row_idx;
            int in_col_idx = out_col_idx + kernel_col_idx;
            int arr_in_idx = (in_row_idx * arr_in_dim) + in_col_idx;

            uint8_t in_val = ((volatile uint8_t*)conv_arr_in)[arr_in_idx];
            int8_t  weight = ((volatile int8_t(*)[CONV_KERNEL_DIM])kernel_w)[kernel_row_idx][kernel_col_idx];

            acc += (int32_t)in_val * (int32_t)weight;
        }
    }

    int arr_out_idx = (out_row_idx * out_dim) + out_col_idx;
    ((volatile uint8_t*)conv_arr_out)[arr_out_idx] = relu_and_descale(acc);
}

//------------------------------------------------------------------------------------------------------------
// Accelerator setup : program the layer parameters and load the kernel once.
//------------------------------------------------------------------------------------------------------------
void conv_xlr_setup(uint8_t* conv_arr_out,
                    uint8_t* conv_arr_in,
                    int      arr_in_dim,
                    int8_t*  kernel_w,
                    int32_t  kernel_b) {

    #ifdef HLCM
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else
    HOST_REG(WGT_ADDR_RI)      = (unsigned int)kernel_w;
    HOST_REG(CONV_BIAS_VAL_RI) = kernel_b;
    HOST_REG(ARR_IN_ADDR_RI)   = (unsigned int)conv_arr_in;
    HOST_REG(ARR_OUT_ADDR_RI)  = (unsigned int)conv_arr_out;
    HOST_REG(ARR_IN_DIM_RI)    = arr_in_dim;

    HOST_REG(XLR_START_RI)     = CONV_SETUP;

    while (!HOST_REG(XLR_DONE_RI)) {
       // poll until the kernel is loaded
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------
// Accelerator window : trigger the HW to compute the ENTIRE output feature-map.
// The FSM loops internally over every row-pair (block A -> row r, block B -> row
// r+half_rows) -- a single host round trip replaces what used to be one round trip
// per row-pair.
//------------------------------------------------------------------------------------------------------------
void conv_window_xlr(void) {

    #ifdef HLCM
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else
    HOST_REG(XLR_START_RI)   = CONV_WINDOW;

    while (!HOST_REG(XLR_DONE_RI)) {
       // poll until the ENTIRE output feature-map has been computed by HW
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------
// Top convolution.
//   Accelerated path : one CONV_SETUP + one CONV_WINDOW round trip total -- the HW
//                      loops all row-pairs internally.
//   Scalar path      : the usual nested per-pixel loop.
//------------------------------------------------------------------------------------------------------------
void conv(uint8_t* conv_arr_out,
          uint8_t* conv_arr_in,
          int      arr_in_dim,
          int8_t*  kernel_w,
          int32_t  kernel_b) {

    #ifdef CONV_XON
    conv_xlr_setup(conv_arr_out, conv_arr_in, arr_in_dim, kernel_w, kernel_b);
    conv_window_xlr();   // HW computes the entire output feature-map internally
    #else
    int out_dim = arr_in_dim - CONV_KERNEL_DIM + 1;
    for (int out_row_idx = 0; out_row_idx < out_dim; out_row_idx++) {
        for (int out_col_idx = 0; out_col_idx < out_dim; out_col_idx++) {
            conv_window_nox(conv_arr_out, conv_arr_in, arr_in_dim,
                            out_row_idx, out_col_idx, kernel_w, kernel_b);
        }
    }
    #endif
}