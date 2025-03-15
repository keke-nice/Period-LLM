import language_evaluation
from pprint import PrettyPrinter
import json

pprint = PrettyPrinter().pprint
evaluator_coco = language_evaluation.CocoEvaluator()

caption_gt_path = './answer_GT.json'
caption_pre_path = './answer_pre.json'

with open(caption_gt_path, 'r') as file:
    caption_gt = json.load(file)

with open(caption_pre_path, 'r') as file:
    caption_pre = json.load(file)

sore_sum = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'CIDEr': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'SPICE': 0.0}

for index in range(len(caption_gt)):
    if caption_pre[index][0] == '':
        caption_pre[index][0] = 'None.'
    try:
        print(f"Evaluating index {index}...")
        print(f"Ground truth: {caption_gt[index][0]}")
        print(f"Prediction: {caption_pre[index][0]}")
        sore = evaluator_coco.run_evaluation(caption_gt[index][0], caption_pre[index][0])
        for key in sore:
            sore_sum[key] += sore[key]
    except BrokenPipeError as e:
        print(f"BrokenPipeError at index {index}: {e}")
        continue  # Skip this iteration and continue with the next one
    except Exception as e:
        print(f"Exception at index {index}: {e}")
        continue  # Skip this iteration and continue with the next one

ave_sore = {}
for key in sore_sum:
    ave_sore[key] = sore_sum[key] / len(caption_gt)

print(ave_sore)
