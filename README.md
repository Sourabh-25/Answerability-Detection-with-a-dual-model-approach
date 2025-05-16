# Answerability Detection Project

This project implements a **dual-model Question Answering (QA) system** using BERT. It consists of:

1. **QA model** - BERT for Answer Span Extraction .
2. **Classifier Model ** - BERT-based Binary Classifier for determining if a question is answerable from the provided context(fine-tuned on SQuAD 2.0).

## 📁 Directory Structure

```
workspace/
├── train_classifier.py   # Train answerability classifier
├── predict.py            # Run dual-model inference
├── models.py             # Model definitions
├── config.yaml           # Configuration file
├── app.py                # prediction with UI
```

## 🔧 Setup

Make sure you have the following installed:

```bash
pip install transformers datasets torch
```
for UI (Optional): 
```bash
pip install Flask
```

## 🏋️‍♀️ Training

Train the binary classifier (answerable vs. not) using SQuAD 2.0:

```bash
python train_classifier.py
```

Note: To save time, it uses only 1% of the dataset. You can increase this in the script.

## 🔍 Inference

Use `predict.py` to run the combined prediction pipeline:

```python
from predict import predict

question = "Who is the president of India?"
context = "Narendra Modi is the prime minister of India."

output = predict(question, context)
print(output)
```

## 🧠 Models Used

- `BERTForQuestionAnswering` from HuggingFace Transformers
- Custom classifier model built on top of `BERTModel` for answerability

## 📌 Notes

- You can swap out BERT with DistilBERT for faster inference.
