import os
import fitz
import json
import re
import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Extract current email from thread ===
def extract_email_data(input_text):
    pattern = r'(On\s.+?<.+?>\swrote:)'
    split_text = re.split(pattern, input_text, maxsplit=1)
    if len(split_text) == 3:
        current_email = split_text[0]
        email_thread = split_text[1] + split_text[2]
        return current_email.strip(), email_thread.strip()
    else:
        print(" Pattern not found — using full email as both current and thread.")
        return input_text.strip(), input_text.strip()

# === PDF Loading & Chunking ===
def load_and_chunk_pdf(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF path is missing or invalid.")
    doc = fitz.open(pdf_path)
    full_text = "".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100) #800/150
    chunks = splitter.split_text(full_text)
    return [Document(page_content=chunk) for chunk in chunks]

# === FAISS Vectorstore ===
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore

def retrieve_relevant_chunks(query, k=4):
    vectorstore = FAISS.load_local(
        "vectorstore",
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True
    )
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

# === GPT-based Response Generator ===
def generate_response(email_thread, relevant_context, model="gpt-3.5-turbo", persona_path="persona.json"):
    with open(persona_path, "r") as f:
        persona = json.load(f)

    context_block = f"\nHere’s the relevant content from the attached PDF:\n{relevant_context}\n" if relevant_context else ""
    context_note = "You are Swagath, responding to an email that includes a PDF attachment."

    prompt = f"""
{context_note}
Your tone is {persona['tone']}. Your goal is to generate a clear, polite, and simple email response in Swagath's voice.

Before writing, do the following:
1. Acknowledge the sender and their message.
2. Identify if the latest message contains any action items or questions.
3. If the attached PDF contains relevant details, briefly incorporate them into the response.
4. If the PDF doesn’t directly inform the response, omit it.
5. Make your reply friendly, concise, and very easy to follow.
6. Avoid repeating the same points or phrases.

Here is the most recent message you are replying to:\n{current_email}
Here’s the full email thread for context:\n{email_thread}

{context_block}

Write a professional and helpful response in Swagath’s writing style. Use some of these phrases when appropriate: {persona['phrases']}

End with:
{persona['signoff']}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )
    return response.choices[0].message.content


# === Logging ===
def log_response(current_email, email_thread, retrieved_context, generated_reply, log_file="response_log.json"):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query_email": current_email,
        "email_thread": email_thread,
        "retrieved_context": retrieved_context,
        "generated_reply": generated_reply
    }

    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=2)
    else:
        with open(log_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2)

# === Main Run ===
if __name__ == "__main__":
    try:
        # Paste full email thread here
        full_thread_text  = """###Paste your full email""" #This will be automated to fetech the email via plugins using Imap

        # Provide PDF path
        pdf_path = "data/myfile.pdf"
        #pdf_path = None
        model_to_use = "gpt-3.5-turbo"  # or "gpt-4"

        current_email, full_thread = extract_email_data(full_thread_text)

        if pdf_path:
            print("\nChunking and indexing PDF...")
            docs = load_and_chunk_pdf(pdf_path)
            create_vectorstore(docs)

            print("\nCurrent email for context search:\n")
            print(current_email[:500] + "...\n")

            
            #query_boost = current_email + " + key details, action items, assignment instructions"
            query_boost = ("Extract relevant information to help respond to this email:\n\n" 
                           + current_email.strip() 
                           +"\n\nThe email may ask for deliverables, clarifications, or deadlines.")

            print("Retrieving relevant content using current message...")
            context = retrieve_relevant_chunks(query_boost)
            #print("Retrieving relevant content using current message...")
            #context = retrieve_relevant_chunks(current_email)
        else:
            print("No PDF path provided — skipping context retrieval.")
            context = None


        print("\nGenerating personalized response...\n")
        reply = generate_response(full_thread, context, model=model_to_use)

        print("\n--- GENERATED EMAIL RESPONSE ---\n")
        print(reply)

        log_response(current_email, full_thread, context, reply)

    except Exception as e:
        print(f"\n Error: {e}")
