import torch
import numpy as np
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Summarizer:
    def __init__(self):
        self.tokenizer = T5TokenizerFast.from_pretrained("paust/pko-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained('SUMMARIZER/sum_model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            rouge_types=['rouge1', 'rouge2', 'rougeL']
        )

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()} 

    def preprocess_logits_for_metrics(self, logits, labels):
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    def preprocess_data(self, text):
        prefix = "summarize: "
        input_text = prefix + text
        model_input = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            padding='max_length'
        )
        return {k: torch.tensor([v]).to(self.device) for k, v in model_input.items()}

    def generate(self, text):
        model_input = self.preprocess_data(text)
        with torch.no_grad():
            outputs = self.model.generate(
                **model_input, 
                output_attentions=True, 
                return_dict_in_generate=True, 
                output_hidden_states=True,
                max_new_tokens=520
            )
        return outputs

    def decode(self, sequences):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]

    def summarize(self, text):
        out = self.generate(text)
        return self.decode(out.sequences), self.calculate_attn(out)

    def calculate_attn(self, outputs):
        cross_attentions = outputs.cross_attentions
        avg_cross_attentions = []

        for attention_per_token in cross_attentions:
            for layer_attention in attention_per_token[-1]:  # Accessing attention tensors from the last layer
                avg_attention_per_head = np.mean(layer_attention.cpu().detach().numpy(), axis=0)
                avg_cross_attentions.append(avg_attention_per_head)

        new_np = np.zeros((1, 1024))

        for obj in avg_cross_attentions:
            new_np = np.vstack((new_np, obj))

        new_np = new_np[1:]  
        attentions = np.mean(new_np, axis=0)
        normalized_attentions = (attentions - np.min(attentions)) / (np.max(attentions) - np.min(attentions))

        return normalized_attentions


    def highlight_text(self, input_tokens, normalized_attentions):
        html_src = "<div>"
        inst = []

        i = 26
        red = (1+i) * 75 % 255
        green = (1+i) * 65 % 255
        blue = (1+i) * 30 % 255

        for j, input_token in enumerate(input_tokens):
            tmp = []
            color_intensity = normalized_attentions[j]
            html_src += f"<span style='background-color: rgba({red}, {green}, {blue}, {color_intensity:.4f})'>{input_token}</span>"
            tmp.append(color_intensity)
        inst.append(tmp)

        return html_src

if __name__ == '__main__':
    # 사용 예시
    precedent = prec

    summarizer = Summarizer()
    summary, attentions = summarizer.summarize(text)

    input_tokens = preprocess_data(precedent)['input_ids'][0]
    input_tokens = list(map(lambda x : tokenizer.decode(x, skip_special_tokens=True), input_tokens.cpu()))

    # prefix 제외
    html_text = summarizer.highlight_text(input_tokens, attentions)
    html_text = html_text+prec[1024-11-14:]
    html_text += "</div>"


    display(summary)
    display()
    for line in html_text.split('\n'):
        display(HTML(line))
