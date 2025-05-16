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
#     print(" Warning: No trained classifier model found. Using untrained QAClassifier.")

# # Move to device
# qa_model.to(device).eval()
# clf_model.to(device).eval()

# # Prediction function
# def predict(question, context, clf_threshold=0.5):
#     inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=384)
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)

#     with torch.no_grad():
#         clf_logits = clf_model(input_ids, attention_mask)
#         clf_probs = F.softmax(clf_logits, dim=1)
#         is_answerable = clf_probs[0][1].item() > clf_threshold

#     if not is_answerable:
#         return {"answerable": False, "answer": None, "confidence": clf_probs[0][1].item()}

#     with torch.no_grad():
#         outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
#         start = torch.argmax(outputs.start_logits)
#         end = torch.argmax(outputs.end_logits)
#         tokens = input_ids[0][start:end + 1]
#         answer = tokenizer.decode(tokens)

#     return {"answerable": True, "answer": answer, "confidence": clf_probs[0][1].item()}



# context='''
# India has a unique culture and is one of the oldest and greatest civilizations of the world. 
# India has achieved all-round socio-economic progress since its Independence. 
# India covers an area of 32,87,263 sq. km, extending from the snow-covered Himalayan heights to the 
# tropical rain forests of the south.
# '''
# question="Is Nepal a good country?"
# print(predict(question,context))

from flask import Flask, render_template, request, redirect, url_for, flash
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
def predict(question, context, clf_threshold=0.999, qa_threshold=0.5, verbose=False):
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
        # return {
            # "answerable": False,
            # "answer": None,
            # "classifier_confidence": clf_probs[0][1].item(),
            # "qa_confidence": None
          return "unanwerable"
        # }

    with torch.no_grad():
        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start = torch.argmax(start_logits, dim=1).item()
        end = torch.argmax(end_logits, dim=1).item()

        if end < start or end - start > 30:
            if verbose:
                print(f"Invalid span detected: start={start}, end={end}")
            # return {
            #     "answerable": False,
            #     "answer": None,
            #     "classifier_confidence": clf_probs[0][1].item(),
            #     "qa_confidence": None
                return "unanwerable"
            # }

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)
        qa_confidence = (start_probs[0][start] + end_probs[0][end]) / 2

        if verbose:
            print(f"QA confidence: {qa_confidence:.4f}")

        if qa_confidence < qa_threshold:
            # return {
            #     "answerable": False,
            #     "answer": None,
            #     "classifier_confidence": clf_probs[0][1].item(),
            #     "qa_confidence": qa_confidence.item()
            # }
            return "unanwerable"
        tokens = input_ids[0][start:end + 1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(answer)
            # return {
        #     "answerable": True,
        #     "answer": answer,
        #     "classifier_confidence": clf_probs[0][1].item(),
        #     "qa_confidence": qa_confidence.item()
        # }
        return answer


app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PANEL1_FILE = 'panel1_messages.txt'
PANEL2_FILE = 'panel2_messages.txt'
ANSWER_FILE = 'answer.txt'

def save_message(filename, message):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(message.strip() + '\n')

def read_answer():
    if os.path.exists(ANSWER_FILE):
        with open(ANSWER_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "No answer available."
    
def read_context():
    if os.path.exists(PANEL1_FILE):
        with open(PANEL1_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "No Context Available."
    
def read_question():
    if os.path.exists(PANEL2_FILE):
        with open(PANEL2_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "No Question Available."

def get_last_message(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else ""
    return ""

# def pdf_to_txt(pdf_path, txt_path):
#     # Open the PDF file in read-binary mode
#     with open(pdf_path, 'rb') as pdf_file:
#         reader = PyPDF2.PdfReader(pdf_file)
#         text = ""

@app.route("/", methods=["GET", "POST"])
def index():
    panel1_message_value = ""
    panel2_message_value = ""

    if request.method == "POST":
        if "pdf_file" in request.files and request.files["pdf_file"].filename != "":
            pdf = request.files["pdf_file"]
            if pdf.filename.endswith(".pdf"):
                pdf.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename))
                flash("PDF uploaded successfully!", "success")
            else:
                flash("Only PDF files are allowed.", "danger")

        elif "panel1_message" in request.form and request.form["panel1_message"].strip():
            panel1_message_value = request.form["panel1_message"]
            save_message(PANEL1_FILE, panel1_message_value)
            save_message(ANSWER_FILE, 'The Answer will appear here.')

        elif "panel2_message" in request.form and request.form["panel2_message"].strip():
            panel2_message_value = request.form["panel2_message"]
            save_message(PANEL2_FILE, panel2_message_value)
            # Load the last saved context from Panel 1
            panel1_message_value = get_last_message(PANEL1_FILE)
            save_message(ANSWER_FILE, (predict(read_question(), read_context(), verbose=False)))


        answer_content = read_answer()

    answer_content = read_answer()

    return render_template(
        "index.html",
        answer_content=answer_content,
        panel1_message_value=panel1_message_value,
        panel2_message_value=panel2_message_value
    )

if __name__ == "__main__":
    app.run(debug=True)


# # Example usage
# context = '''
# India has a unique culture and is one of the oldest and greatest civilizations of the world. 
# India has achieved all-round socio-economic progress since its Independence.
#  India covers an area of 32,87,263 sq. km,
#  extending from the snow-covered Himalayan heights to the tropical rain forests of the south.
# '''
# # question = "Is India a good country?"

# # result = predict(question, context, verbose=True)
# # print(result)
