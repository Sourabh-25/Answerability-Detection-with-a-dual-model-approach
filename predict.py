

from transformers import BertTokenizer, BertForQuestionAnswering
from models import QAClassifier
import torch
import torch.nn.functional as F
import os

# Load models and tokenizer
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier with weights
clf_model = QAClassifier()
if os.path.exists("qa_classifier_model.pt"):
    clf_model.load_state_dict(torch.load("qa_classifier_model.pt", map_location=device))
    print("Loaded trained QAClassifier model.")
else:
    print("Warning: No trained classifier model found. Using untrained QAClassifier.")

# Move to device
qa_model.to(device).eval()
clf_model.to(device).eval()

# Prediction function
def predict(question, context, clf_threshold=0.5, qa_threshold=0.3, verbose=False):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=384)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        clf_logits = clf_model(input_ids, attention_mask)
        clf_probs = F.softmax(clf_logits, dim=1)
        is_answerable = clf_probs[0][1].item() > clf_threshold

    if verbose:
        print(f"Classifier probability (answerable): {clf_probs[0][1].item():.4f}")

    if not is_answerable:
        return {
            "answerable": False,
            "answer": None,
            "classifier_confidence": clf_probs[0][1].item(),
            "qa_confidence": None
        }

    with torch.no_grad():
        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start = torch.argmax(start_logits, dim=1).item()
        end = torch.argmax(end_logits, dim=1).item()

        if end < start or end - start > 30:
            if verbose:
                print(f"Invalid span detected: start={start}, end={end}")
            return {
                "answerable": False,
                "answer": None,
                "classifier_confidence": clf_probs[0][1].item(),
                "qa_confidence": None
            }

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)
        qa_confidence = (start_probs[0][start] + end_probs[0][end]) / 2

        if verbose:
            print(f"QA confidence: {qa_confidence:.4f}")

        if qa_confidence < qa_threshold:
            return {
                "answerable": False,
                "answer": None,
                "classifier_confidence": clf_probs[0][1].item(),
                "qa_confidence": qa_confidence.item()
            }

        tokens = input_ids[0][start:end + 1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return {
            "answerable": True,
            "answer": answer,
            "classifier_confidence": clf_probs[0][1].item(),
            "qa_confidence": qa_confidence.item()
        }

# Example usage
context = '''
India has a unique culture and is one of the oldest and greatest civilizations of the world. India has achieved all-round socio-economic progress since its Independence. India covers an area of 32,87,263 sq. km, extending from the snow-covered Himalayan heights to the tropical rain forests of the south.
'''
question = "which is that?"

result = predict(question, context, verbose=True)
print(result)
