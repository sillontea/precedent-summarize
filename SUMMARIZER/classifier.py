# 문장 분류를 위한 준비
# setting for sentence classification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os 

class Classifier:
    def __init__(self):
        model_path = os.path.join(os.getcwd(),"SUMMARIZER", "model")
        # classification labels
        self.labels = {'01_주장':0, '02_사실':1, '03_판단':2, '04_결론':3}
        self.n_labels = len(self.labels)
        self.lb2txt = {0: '<주장>', 1: '<사실>', 2: '<판단>', 3: '<결론>'}

        # model은 model_path에 파인튜닝 된 가중치가 위치해야 합니다.
        # 학습 시 이 부분에 학습되지 않은 모델을 가져와도 됩니다.
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) # model_path "beomi/kobert"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=self.n_labels)
        self.model.to(self.device)

        self.model.resize_token_embeddings(len(self.tokenizer))
        
        
    def infer(self, text):
        batch = self.preprocessing([text])
        result = self.inference(batch)
        return result

    def preprocessing(self, sequences:list):
        # Get all texts from sequences list.
        texts = [sequence for sequence in sequences]
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=None)
        return inputs

    def inference(self, batch):
        device = self.device
        model = self.model
        
        model.eval()
        model.to(device)
        batch.to(device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

        return predict_content

    
# 사실 패턴 유무를 확인하기 위한 함수
# setting for checking fact pattern
import re
from IPython.display import display, HTML

def highlight_fact_pattern(sentence, width=90):
    view_text = ""
    pattern = re.compile(r"사실")

    last_end = 0
    for i, obj in enumerate(pattern.finditer(sentence)):
        start, end = obj.span()
        view_text += sentence[last_end:start]
        view_text += ''.join(["<span style='color:red';>", sentence[start:end], "</span>"])
        last_end = end
    view_text += sentence[last_end:]

    return view_text

def check_fact_pattern(sentence):
    check = re.search(r"사실", sentence)
    return True if check != None else False

def draw_line():
    print('====='*20)

