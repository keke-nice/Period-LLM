import json
import re
import numpy as np

# 英文数字单词到阿拉伯数字的映射
english_to_arabic = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'twice': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20
}
below10 = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'twice': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9
}
num_10_20 = {
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20
}
# 定义一个函数来提取次数在10以下的文件
def extract_below10(your_json_file, save_path):
    with open(your_json_file, 'r') as file:
        data = json.load(file)
    save_list = []
    for item in data:
        video_path = item['video']
        Question = item['QA']['q']
        # Answer = item['QA']['a_GT']
        Answer = item['QA']['a']
        arabic_number = re.findall(r'\d+', Answer)
        if len(arabic_number) >= 1:
            for i, number in enumerate(arabic_number):
                # if (int(number) >= 10) & (int(number)<= 20):
                if (int(number) < 10):
                  save_list.append(item)
                  break
        else: 
            flag = 0
            for word in Answer.split():
              word = re.sub(r'[,.\s]+$', '', word)
              # if word.lower() in num_10_20:
              if word.lower() in below10:
                  # 将英文数字单词转换为阿拉伯数字并添加到列表中
                  save_list.append(item)
                  flag = 1

                  break
    with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_list, f, ensure_ascii=False, indent=4)
      
    

# 定义一个函数来提取数字
def extract_numbers(your_json_file):
    with open(your_json_file, 'r') as file:
        data = json.load(file)
    numbers = []
    # 遍历数据中的每个项目（在这里是一个列表的列表）
    total = len(data)
    for item in data:
        # 遍历项目中的每个字符串
        for sentence in item:
            # 使用正则表达式匹配阿拉伯数字
            arabic_number = re.findall(r'\d+', sentence)
            if len(arabic_number) >= 1:
                for i, number in enumerate(arabic_number):
                    if int(number) < 100:
                      numbers.append(int(number))
                      break
                    if i==len(arabic_number)-1:
                        numbers.append(int(0))
            else: 
                flag = 0
                for word in sentence.split():
                  word = re.sub(r'[,.\s]+$', '', word)
                  if word.lower() in english_to_arabic:
                      # 将英文数字单词转换为阿拉伯数字并添加到列表中
                      numbers.append(english_to_arabic[word.lower()])
                      flag = 1
                      break
                if flag == 0:
                    numbers.append(int(0))
    assert len(numbers) == total
    print(numbers)
    return numbers

def MyEval(numbers_pr, numbers_rel):
    numbers_pr = np.array(numbers_pr).reshape(-1)
    numbers_rel = np.array(numbers_rel).reshape(-1)
    temp = numbers_pr - numbers_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / (numbers_rel + 1))
    p = np.sum((numbers_pr - np.mean(numbers_pr))*(numbers_rel - np.mean(numbers_rel))) / (
                0.01 + np.linalg.norm(numbers_pr - np.mean(numbers_pr), ord=2) * np.linalg.norm(numbers_rel - np.mean(numbers_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p

if __name__ == '__main__':
    # 提取数字
    GT_numbers = extract_numbers('./answer_GT.json')
    pre_numbers = extract_numbers('./answer_pre.json')
    MyEval(GT_numbers, pre_numbers)
    # extract_below10('./Result/infer_train_double_Label_countix_textframe100margin1200000/QA.json', './Result/infer_train_double_Label_countix_textframe100margin1200000/QA_10_20.json')
    # extract_below10('./countix/number_countix_val.json', './countix/number_countix_val_below10.json')
