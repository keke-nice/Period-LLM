git export CUDA_VISIBLE_DEVICES=6
python train.py \
--json  "./countix/number_countix_train.json" \
--image_size 224 \
--visual 'clip' \
--job_name 'hook_dynamic_countix_frame5s_base_on_text' \
--frame_num 20 \
--ckpts './result_Model_new/text_original200000.pth' \
--input_type 'video' \
--max_iter 200000 \
--learning-rate 0.001


