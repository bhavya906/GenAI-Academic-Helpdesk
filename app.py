from flask import Flask, request, jsonify, render_template
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import pipeline

print("🚀 Help Desk Starting...")

app = Flask(__name__)

# -----------------------------
# Load LLM
# -----------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=150,
)

llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are a College Academic Assistant.

Answer ONLY using the given context.

If the answer is not available in the context, reply:

"I couldn't find that information in the uploaded documents."

Give only a short answer.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# -----------------------------
# Embedding Model
# -----------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create QA for every PDF
# -----------------------------

qa_systems = {}

pdf_folder = "uploads"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        docs = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            docs,
            embedding
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k":1}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt":prompt
            },
            return_source_documents=True
        )

        qa_systems[file.lower()] = qa

print("Loaded PDFs :", list(qa_systems.keys()))

# -----------------------------
# Student Data
# -----------------------------

students = {

    "john":"John - CSE - 3rd Year - 9876543210",

    "anita":"Anita - ECE - 2nd Year - 9123456780",

    "rahul":"Rahul - ME - 4th Year - 9988776655"

}

# -----------------------------
# Home
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Ask
# -----------------------------

@app.route("/ask",methods=["POST"])
def ask():

    data=request.get_json()

    if not data:
        return jsonify({"answer":"Ask a question."})

    question=data["question"].lower()

    # Greetings

    if question in ["hello","hi","hey"]:
        return jsonify({"answer":"Hello 👋 Welcome to GenAI Academic Helpdesk."})

    if "thank" in question:
        return jsonify({"answer":"You're welcome 😊"})

    if "bye" in question:
        return jsonify({"answer":"Goodbye 👋"})

    # Student Search

    for word in question.split():

        if word in students:

            return jsonify({"answer":students[word]})

    # -----------------------------
    # Select PDF
    # -----------------------------

    if "hostel" in question:
        qa = qa_systems.get("hostel.pdf")

    elif "library" in question or "college" in question:
        qa = qa_systems.get("college information system.pdf")

    elif "placement" in question:
         qa = qa_systems.get("placement.pdf")

    elif "exam" in question or "circular" in question:
         qa = qa_systems.get("circular.pdf")

    elif "syllabus" in question:
        qa = qa_systems.get("syllabus.pdf")

    else:
         return jsonify({
        "answer": "Please ask about Hostel, Library, Placement, Exam or Syllabus."
    })

    if qa is None:
        return jsonify({
        "answer": "The required PDF could not be found."
    })
    print("Loaded PDFs :", list(qa_systems.keys()))
    # -----------------------------
    # Generate Answer
    # -----------------------------

    try:

        result=qa.invoke({"query":question})

        answer=result["result"]

        doc=result["source_documents"][0]

        source=os.path.basename(doc.metadata["source"])

        page=doc.metadata.get("page",0)

        return jsonify({

            "answer":answer,

            "source":source,

            "page":page

        })

    except Exception as e:

        print(e)

        return jsonify({

            "answer":"Sorry, something went wrong."

        })

# -----------------------------
# Run
# -----------------------------

if __name__=="__main__":
    app.run(debug=True)