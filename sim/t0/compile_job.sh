#!/bin/bash
export LIBS_PATH=$K5_LIBS
shopt -s expand_aliases
source $K5_XBOX_ENV/setup/k5_rc3_setup.sh
export K5_APP_SHARED_LIB="/project/tsmc65/shared/k5_share/slrx_ref/sw/apps/slr_shared"
cd $K5_SW_APPS/slrx 
rm -rf build/* 
$K5_ENV/sw/sw_utils/comp_app_local_rc3.sh slrx _SPMT_  "-D_NUM_ITR_=1 -DALL_XON -D_XBOX_ -I$K5_XBOX_ENV/sw/xbox_libs" _MAX10_FPGA_ 24576
cd -
