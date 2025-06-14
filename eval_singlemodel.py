import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from datasets import load_dataset
from tqdm import tqdm
import string
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LOG_FILE = "bert_qa_logs.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(device).eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Normalization functions
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

def compute_exact(a_pred, a_true):
    return int(normalize_text(a_pred) == normalize_text(a_true))

def compute_f1(pred, truth):
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

# Dataset
data = load_dataset("squad_v2", split="validation[:300]")

exact_scores = []
f1_scores = []
y_true = []
y_pred = []

with open(LOG_FILE, "w", encoding="utf-8") as f:
    for example in tqdm(data, desc="Evaluating"):
        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]
        is_answerable = 1 if len(answers) > 0 else 0
        y_true.append(is_answerable)

        inputs = tokenizer(question, context, return_tensors="pt",
                           truncation="only_second", padding="max_length",
                           max_length=384).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits)

        if start > end or end - start > 30:
            pred_answer = ""
        else:
            tokens = inputs["input_ids"][0][start:end+1]
            pred_answer = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pred_is_answerable = 0 if pred_answer.strip() == "" else 1
        y_pred.append(pred_is_answerable)

        # EM/F1
        if is_answerable:
            em = max(compute_exact(pred_answer, a) for a in answers)
            f1_val = max(compute_f1(pred_answer, a) for a in answers)
        else:
            em = 1 if pred_answer.strip() == "" else 0
            f1_val = em

        exact_scores.append(em)
        f1_scores.append(f1_val)

        f.write("="*80 + "\n")
        f.write(f"Question            : {question}\n")
        f.write(f"Predicted Answer    : {pred_answer}\n")
        f.write(f"Ground Truth Answer : {answers}\n")
        f.write(f"True Answerable     : {is_answerable} | Predicted: {pred_is_answerable}\n")
        f.write(f"Exact Match         : {em} | F1 Score: {f1_val:.4f}\n")

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_binary, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("\n=== Evaluation on SQuAD v2 (Standard BERT QA) ===")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score (binary) : {f1_binary:.4f}")
print(f"Exact Match (EM)  : {sum(exact_scores)/len(exact_scores):.4f}")
print(f"F1 Score (token)  : {sum(f1_scores)/len(f1_scores):.4f}")
print(f"Logs saved to     : {LOG_FILE}")
