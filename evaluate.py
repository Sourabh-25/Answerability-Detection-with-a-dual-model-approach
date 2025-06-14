# import json
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# import torch
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from transformers import BertTokenizer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from tqdm import tqdm

# # Tokenizer used during dataset preparation
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Dataset class


# from transformers import BertTokenizer, BertForQuestionAnswering
# from models import QAClassifier
# import torch
# import torch.nn.functional as F
# import os

# # Load models and tokenizer
# qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load classifier with weights
# clf_model = QAClassifier()
# if os.path.exists("qa_classifier_model.pt"):
#     clf_model.load_state_dict(torch.load("qa_classifier_model.pt", map_location=device))
#     print("Loaded trained QAClassifier model.")
# else:
#     print("Warning: No trained classifier model found. Using untrained QAClassifier.")

# # Move to device
# qa_model.to(device).eval()
# clf_model.to(device).eval()


# class AnswerableDataset(torch.utils.data.Dataset):
#     def __init__(self, split='validation'):
#         data = load_dataset('squad_v2', split=split)
#         self.samples = []
#         self.raw = []  # store original question/context for predict()

#         for item in data:
#             inputs = tokenizer(item['question'], item['context'],
#                                truncation='only_second',
#                                padding='max_length',
#                                max_length=384,
#                                return_tensors='pt')
#             label = 0 if len(item['answers']['text']) == 0 else 1
#             self.samples.append((item['question'], item['context'], label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]
# def evaluate(dataset, clf_threshold=0.999, qa_threshold=0.5, batch_size=4, limit=None):
#     if limit:
#         dataset = torch.utils.data.Subset(dataset, range(limit))

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     y_true = []
#     y_pred = []

#     print(f"\n{'='*20} Evaluation Start {'='*20}\n")

#     for batch in tqdm(loader):
#         questions, contexts, labels = batch

#         inputs = tokenizer(list(questions), list(contexts),
#                            truncation=True,
#                            padding="max_length",
#                            max_length=384,
#                            return_tensors="pt")

#         input_ids = inputs['input_ids'].to(device)
#         attention_mask = inputs['attention_mask'].to(device)

#         with torch.no_grad():
#             clf_logits = clf_model(input_ids, attention_mask)
#             clf_probs = F.softmax(clf_logits, dim=1)
#             is_answerable_batch = clf_probs[:, 1] > clf_threshold

#         for i in range(len(questions)):
#             q = questions[i]
#             c = contexts[i]
#             true_label = labels[i].item()
#             pred_label = 0
#             classifier_confidence = clf_probs[i][1].item()

#             print(f"\nQuestion #{len(y_true)+1}: {q}")
#             print(f"True Label      : {'Answerable' if true_label else 'Unanswerable'}")
#             print(f"Classifier Conf.: {classifier_confidence:.4f}")

#             if not is_answerable_batch[i]:
#                 print(f"Predicted       : Unanswerable (by classifier)")
#                 y_pred.append(0)
#                 y_true.append(true_label)
#                 continue

#             single_input_ids = input_ids[i].unsqueeze(0)
#             single_attention_mask = attention_mask[i].unsqueeze(0)

#             with torch.no_grad():
#                 outputs = qa_model(input_ids=single_input_ids, attention_mask=single_attention_mask)
#                 start_logits = outputs.start_logits
#                 end_logits = outputs.end_logits

#                 start = torch.argmax(start_logits, dim=1).item()
#                 end = torch.argmax(end_logits, dim=1).item()

#                 if end < start or end - start > 30:
#                     print("Predicted       : Unanswerable (invalid span)")
#                     y_pred.append(0)
#                     y_true.append(true_label)
#                     continue

#                 start_probs = F.softmax(start_logits, dim=1)
#                 end_probs = F.softmax(end_logits, dim=1)
#                 qa_confidence = (start_probs[0][start] + end_probs[0][end]) / 2

#                 print(f"QA Confidence   : {qa_confidence:.4f}")

#                 if qa_confidence < qa_threshold:
#                     print("Predicted       : Unanswerable (low QA confidence)")
#                     y_pred.append(0)
#                     y_true.append(true_label)
#                     continue

#                 answer_tokens = input_ids[i][start:end + 1]
#                 answer = tokenizer.decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

#                 print(f"Predicted       : Answerable")
#                 print(f"Extracted Answer: {answer}")

#                 y_pred.append(1)
#                 y_true.append(true_label)

#     acc = accuracy_score(y_true, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

#     print(f"\n{'='*20} Evaluation Metrics {'='*20}")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"Precision : {precision:.4f}")
#     print(f"Recall    : {recall:.4f}")
#     print(f"F1 Score  : {f1:.4f}")
#     print(f"{'='*58}")

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     dataset = AnswerableDataset(split="validation")
#     evaluate(dataset, clf_threshold=0.999, qa_threshold=0.5, limit=100)  # Evaluate on 500 examples for speed

import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering
from models import QAClassifier
import torch.nn.functional as F
import os


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
clf_model = QAClassifier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_model.to(device).eval()

if os.path.exists("qa_classifier_model.pt"):
    clf_model.load_state_dict(torch.load("qa_classifier_model.pt", map_location=device))
    print("Loaded trained QAClassifier model.")
else:
    print("Warning: No trained classifier model found. Using untrained QAClassifier.")

clf_model.to(device).eval()


class AnswerableDataset(Dataset):
    def __init__(self, split='validation'):
        data = load_dataset('squad_v2', split=split)
        self.samples = []
        for item in data:
            label = 0 if len(item['answers']['text']) == 0 else 1
            # Store the first gold answer if available
            answer = item['answers']['text'][0] if label == 1 else ""
            self.samples.append((item['question'], item['context'], label, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def qa_collate_fn(batch):
    questions, contexts, labels, answers = zip(*batch)
    return list(questions), list(contexts), torch.tensor(labels), list(answers)


def evaluate(dataset, clf_threshold=0.999, qa_threshold=0.5, batch_size=4, limit=None):
    if limit:
        dataset = torch.utils.data.Subset(dataset, range(limit))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=qa_collate_fn)

    y_true_cls = []
    y_pred_cls = []

    total_em = 0
    total_answerable = 0

    print(f"\n{'='*20} Evaluation Start {'='*20}\n")

    for batch in tqdm(loader):
        questions, contexts, labels, true_answers = batch

        inputs = tokenizer(questions, contexts,
                           truncation=True,
                           padding="max_length",
                           max_length=384,
                           return_tensors="pt")

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            clf_logits = clf_model(input_ids, attention_mask)
            clf_probs = F.softmax(clf_logits, dim=1)
            is_answerable_batch = clf_probs[:, 1] > clf_threshold

        for i in range(len(questions)):
            q = questions[i]
            c = contexts[i]
            true_label = labels[i].item()
            gold_answer = true_answers[i]
            pred_label = 0
            classifier_confidence = clf_probs[i][1].item()

            print(f"\nQuestion #{len(y_true_cls)+1}: {q}")
            print(f"True Label      : {'Answerable' if true_label else 'Unanswerable'}")
            print(f"Classifier Conf.: {classifier_confidence:.4f}")

            if not is_answerable_batch[i]:
                print(f"Predicted       : Unanswerable (by classifier)")
                y_pred_cls.append(0)
                y_true_cls.append(true_label)
                continue

            single_input_ids = input_ids[i].unsqueeze(0)
            single_attention_mask = attention_mask[i].unsqueeze(0)

            with torch.no_grad():
                outputs = qa_model(input_ids=single_input_ids, attention_mask=single_attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                start = torch.argmax(start_logits, dim=1).item()
                end = torch.argmax(end_logits, dim=1).item()

                if end < start or end - start > 30:
                    print("Predicted       : Unanswerable (invalid span)")
                    y_pred_cls.append(0)
                    y_true_cls.append(true_label)
                    continue

                start_probs = F.softmax(start_logits, dim=1)
                end_probs = F.softmax(end_logits, dim=1)
                qa_confidence = (start_probs[0][start] + end_probs[0][end]) / 2

                print(f"QA Confidence   : {qa_confidence:.4f}")

                if qa_confidence < qa_threshold:
                    print("Predicted       : Unanswerable (low QA confidence)")
                    y_pred_cls.append(0)
                    y_true_cls.append(true_label)
                    continue

                answer_tokens = input_ids[i][start:end + 1]
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                print(f"Predicted       : Answerable")
                print(f"Extracted Answer: {answer}")
                print(f"Gold Answer     : {gold_answer}")

                y_pred_cls.append(1)
                y_true_cls.append(true_label)

                # Compute EM
                if true_label == 1:
                    total_answerable += 1
                    if normalize_answer(answer) == normalize_answer(gold_answer):
                        total_em += 1

    acc = accuracy_score(y_true_cls, y_pred_cls)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average='binary')
    em_score = total_em / total_answerable if total_answerable > 0 else 0.0

    print(f"\n{'='*20} Evaluation Metrics {'='*20}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Exact Match  : {em_score:.4f}")
    print(f"{'='*58}")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    dataset = AnswerableDataset(split="validation")
    evaluate(dataset, clf_threshold=0.999, qa_threshold=0.5, limit=300)
