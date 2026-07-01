#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h"

//------------------------------------------------------------------------------------------------------------
// Highly optimized Select / ArgMax for SLR output vector.
// Assumes NUM_LABELS = 27.
// Fully unrolled: no loop overhead, no vec_size dependency.
// Values are uint8_t, not int8_t.

static inline int __attribute__((always_inline)) get_max_val_idx_27(uint8_t* vec_in) {

    uint8_t* v = vec_in;

    int max_val_idx = 0;
    uint8_t max_val = v[0];

#define CHECK_LABEL(IDX)                 \
    do {                                 \
        uint8_t val = v[(IDX)];          \
        if (val > max_val) {             \
            max_val = val;               \
            max_val_idx = (IDX);         \
        }                                \
    } while (0)

    CHECK_LABEL(1);
    CHECK_LABEL(2);
    CHECK_LABEL(3);
    CHECK_LABEL(4);
    CHECK_LABEL(5);
    CHECK_LABEL(6);
    CHECK_LABEL(7);
    CHECK_LABEL(8);
    CHECK_LABEL(9);
    CHECK_LABEL(10);
    CHECK_LABEL(11);
    CHECK_LABEL(12);
    CHECK_LABEL(13);
    CHECK_LABEL(14);
    CHECK_LABEL(15);
    CHECK_LABEL(16);
    CHECK_LABEL(17);
    CHECK_LABEL(18);
    CHECK_LABEL(19);
    CHECK_LABEL(20);
    CHECK_LABEL(21);
    CHECK_LABEL(22);
    CHECK_LABEL(23);
    CHECK_LABEL(24);
    CHECK_LABEL(25);
    CHECK_LABEL(26);

#undef CHECK_LABEL

    return max_val_idx;
}

//------------------------------------------------------------------------------------------------------------
// Generic fallback, kept for compatibility.
// The real SLR path below calls get_max_val_idx_27 directly.

int get_max_val_idx(uint8_t* vec_in, int vec_size) {

    if (vec_size == NUM_LABELS) {
        return get_max_val_idx_27(vec_in);
    }

    int max_val_idx = 0;
    uint8_t max_val = vec_in[0];

    for (int i = 1; i < vec_size; i++) {
        uint8_t val = vec_in[i];

        if (val > max_val) {
            max_val = val;
            max_val_idx = i;
        }
    }

    return max_val_idx;
}

//============================================================================================================

int infer(uint8_t slr_img [IMG_DIM][IMG_DIM],      // inferred image
          slr_model_params_t* slr_model_params_p,  // model parameters
          slr_intr_fm_t*      slr_intr_fm_p,       // intermediate feature-maps
          char                check_performance,   // enable performance monitoring
          int                 img_idx) {           // inferred image index, just for reporting  
         
     
        if (check_performance) reset_report_performance();
     
        // Perform Inference

        // Calling CONV0 Layer
        conv((uint8_t*)(slr_intr_fm_p->post_c0_fm),         // Conv0 output feature-map
             (uint8_t*) slr_img,                            // Conv0 input image  
                        IMG_DIM,                            // Conv0 input dimensions      
             (int8_t*) (slr_model_params_p->conv0_w),       // Conv0 kernel weights
             (int32_t) (slr_model_params_p->conv0_b));      // Conv0 kernel bias
             
        if (check_performance) report_task_performance("Conv0");              
       
        // Calling POOL0 Layer
        pool_max_2x2((uint8_t*)(slr_intr_fm_p->post_p0_fm), // Pool0 output feature-map
                     (uint8_t*)(slr_intr_fm_p->post_c0_fm), // Pool0 input image <- output of Conv0
                     POST_C0_FM_DIM);                       // Pool0 input dimensions <- post Conv0

        if (check_performance) report_task_performance("Pool0");      
       
        // Calling CONV1 Layer
        conv((uint8_t*)(slr_intr_fm_p->post_c1_fm),         // Conv1 output feature-map <- Pool0 output feature-map
             (uint8_t*)(slr_intr_fm_p->post_p0_fm),         // Conv1 input image  
                        POST_P0_FM_DIM,                     // Conv1 input dimensions      
             (int8_t*) (slr_model_params_p->conv1_w),       // Conv1 kernel weights
             (int32_t) (slr_model_params_p->conv1_b));      // Conv1 kernel bias

        if (check_performance) report_task_performance("Conv1");  
       
        // Calling POOL1 Layer
        pool_max_2x2((uint8_t*)(slr_intr_fm_p->post_p1_fm), // Pool1 output feature-map
                     (uint8_t*)(slr_intr_fm_p->post_c1_fm), // Pool1 input image <- output of Conv1
                     POST_C1_FM_DIM);                       // Pool1 input dimensions <- post Conv1

        if (check_performance) report_task_performance("Pool1");  
       
        // Calling LINEAR0 Layer
        linear((uint8_t*)(slr_intr_fm_p->post_lin0_fm),     // Linear0 output feature-map, single row
               (uint8_t*)(slr_intr_fm_p->post_p1_fm),       // Linear0 input vector <- output of Pool1, flat
               LIN_INVEC_SIZE,                              // Linear0 input dimension
               LIN_HID_DIM,                                 // Linear0 output dimension              
               (int8_t*)(slr_model_params_p->lin0_w_trn),   // Linear0 weights transposed
               slr_model_params_p->lin0_b);                 // Linear0 bias

        if (check_performance) report_task_performance("Linear0");  
       
        // Calling LINEAR1 Layer
        linear((uint8_t*)(slr_intr_fm_p->post_lin1_fm),     // Linear1 output feature-map, single row
               (uint8_t*)(slr_intr_fm_p->post_lin0_fm),     // Linear1 input vector <- output of Linear0
               LIN_HID_DIM,                                 // Linear1 input dimension
               NUM_LABELS,                                  // Linear1 output dimension            
               (int8_t*)(slr_model_params_p->lin1_w_trn),   // Linear1 weights transposed
               slr_model_params_p->lin1_b);                 // Linear1 bias

        if (check_performance) report_task_performance("Linear1");  
 
        // Final label selection - optimized fully-unrolled ArgMax for 27 labels
        int detected_label_idx = get_max_val_idx_27((uint8_t*)(slr_intr_fm_p->post_lin1_fm));

        if (check_performance) {
          report_task_performance("Select");
          report_total_performance();
          printf("\n");
        }
                       
        return detected_label_idx;  
}  

//====================================================================================================

int main() {

  bm_printf("\n\nHELLO K5X SLR : Sign Language Recognition\n\n");
 
  alloc_init(); // Initialize XMEM memory allocator
 
  slr_model_params_t* slr_model_params_p = load_model_params();
 
  // Allocating layers intermediate feature-maps    
  int fm_total_num_bytes = sizeof(slr_intr_fm_t);
  printf("Allocated total %d bytes of for intermediate feature -maps\n", fm_total_num_bytes);
  slr_intr_fm_t* slr_intr_fm_p = (slr_intr_fm_t*)alloc_get(fm_total_num_bytes, "slr_intr_fm");              
   
  int num_imgs_in_buf = 1; // Currently we load a single image per iteration
  int imgs_total_num_bytes = num_imgs_in_buf * sizeof(slr_ds_image_t);
 
  slr_ds_image_t* slr_ds_imgs = (slr_ds_image_t*)alloc_get(imgs_total_num_bytes, "slr_ds_imgs");
 
  char* ds_test_file_path = "$K5_SHARE/slrx_ref/sw/apps/slr_shared/pt/workspace/slr_ds_mnx.txt";  
 
  FILE_REF file_ref; // assigned by load_hex_file(... OPEN*)  
 
  printf("\n\n");
 
  for (int img_idx = 0; img_idx < NUM_TEST_IMAGES; img_idx++) {

    file_access_mode_t file_access_mode = img_idx == 0 ? OPEN : CONT;
    load_hex_file(ds_test_file_path, &file_ref, (char*)slr_ds_imgs, imgs_total_num_bytes, file_access_mode); // Keep open
 
    char check_performance = (NUM_TEST_IMAGES == 1); // check performance on single image invocation
    char is_last_ds_img    = slr_ds_imgs[0].is_last_img || (NUM_TEST_IMAGES == 1);
    int  expected_label_id = slr_ds_imgs[0].slr_img_label_id; // Currently buffer holds one image
       
    int detected_label_id = infer(slr_ds_imgs[0].slr_img, // inferred image
                                  slr_model_params_p,     // model parameters
                                  slr_intr_fm_p,          // intermediate feature-maps memory space
                                  check_performance,      // enable performance check
                                  img_idx);               // running image index
         
    char text_mode = TRUE; // opposed to dataset benchmarking
    output_detection(detected_label_id, expected_label_id, img_idx, is_last_ds_img, text_mode);    

    if (is_last_ds_img || ((img_idx + 1) >= NUM_TEST_IMAGES)) break;
  }
 
  fclose(file_ref);
     
  // Free all allocated memory space
 
  alloc_free((void*)slr_model_params_p, "model_params");    
  alloc_free((void*)slr_ds_imgs,        "slr_ds_imgs");
  alloc_free((void*)slr_intr_fm_p,      "slr_intr_fm");
 
  if (NUM_TEST_IMAGES == 1) printf("\n\nDone Single Detection With Performance Check\n");

  bm_quit_app();
  return 0;
}