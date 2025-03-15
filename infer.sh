export CUDA_VISIBLE_DEVICES=2
python infer.py \
--infer_model "./result_Model_new/hook_dynamic_countix_frame5s_base_on_text200000.pth" \
--json './countix/number_countix_val.json' \
--input_type 'video' \
--frame_num 20 
python evaluation.py
