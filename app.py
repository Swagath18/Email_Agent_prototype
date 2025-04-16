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
        full_thread_text  = """Hello Swagath,
We hope this note finds you well!
It has been a pleasure meeting you and learning about your career journey!

As discussed, the next step in our ongoing exploration is a Take-home Assignment (attached). 
Kindly review the attached document and let us know if you have any clarifying questions. We have deliberately structured the assignment to not be too prescriptive, so you may determine the amount of time and effort investment in creating a response. You are welcome to use all available tools to complete this assignment provided you transparently outline what they are in your presentation (see attachment)  
Please email your response on this thread, latest by midnight, Thu-April 17, 2025. 
Do let us know if for any reason you can't respond by this deadline.

If your submission is shortlisted, you will be invited for a 90-minute in-person Round-2 interview in the SF Bay Area, which will primarily focus on:
   i> Assessment of your hands-on system design and software engineering capability (DSA), to build and deliver AI driven enterprise products.
   i> Dive deeper into your take-home assignment submission.
   (note: use of any technology tools or aids shall not be permitted during the Round-2 interview)


Thanks again for your interest in our venture!

The Hiring Team @ Ease Vertical AI!
www.easeverticalai.com


On Sat, Apr 12, 2025 at 10:55 AM SWAGATH BABU <swagathb18@g.ucla.edu> wrote:
Hi Vivek,

Hope you had a great week! 

I wanted to check in and see if there’s any update on the next steps for the Applied ML Engineer role. I really enjoyed our chat last week and I’m still super excited about the opportunity to be part of what you’re building at Ease AI.

Just wanted to see if there’s any update on the assessment round or timelines moving forward. Happy to provide anything else you might need from my end.  
Looking forward to hearing from you!


On Mon, Apr 7, 2025 at 5:24 PM SWAGATH BABU <swagathb18@g.ucla.edu> wrote:
Hi Vivek,
Thank you for the opportunity to discuss my background and approach to the problem. It was really great to talk to you, I realized that I initially introduced a graph-based approach for the recommendation system, which I believe is a strong fit for modeling social relationships. Later, I also mentioned using tries for certain features, such as handling user data like interests, locations, and organizations. While tries can be useful for some specific tasks, I now recognize that a graph-based structure is a more natural and optimal solution for the core recommendation logic. 

I’m really excited about the opportunity to contribute to Ease AI. I’m confident that my experience in machine learning, deep learning, and MLOps pipelines would allow me to make a meaningful impact on your team’s work.

Additionally, I wanted to mention that I am currently on F1 OPT and would require E-Verify for employment for F1 STEM OPT. Please let me know if you need any further information on this and for your reference here is the resource (https://studyinthestates.dhs.gov/students/work/understanding-e-verify)

Thank you again for your time, and I look forward to the next steps.

Best regards,
Swagath Babu


On Wed, Apr 2, 2025 at 4:14 PM SWAGATH BABU <swagathb18@g.ucla.edu> wrote:
Hi ,  
Thanks for the update! I completely understand. I’d be happy to move the call to Monday at 2 PM PST that works well for me. 
It would be great if you can send an updated invite. 

Thank you and I'm looking forward to the conversation.


On Wed, Apr 2, 2025 at 3:58 PM Engage @easeverticalai <Engage@easeverticalai.com> wrote:
Hello Swagath,

We apologise, but due to a last minute trip that our CEO needs to make to Southern California, we’ll need to reschedule today’s call.

We currently have two rescheduling options:
- Our CEO could meet you in person in the Santa Monica area around 11am tomorrow.
- Or, we could move today’s call to 2pm PST on Monday.

Kindly confirm if either of these options work for you, and the one you’d prefer.


Thanks again for your interest in our venture!

The Hiring Team @ Ease Vertical AI!
www.easeverticalai.com



On Mon, Mar 31, 2025 at 4:23 PM SWAGATH BABU <swagathb18@g.ucla.edu> wrote:
Hi,

Yes sure it works for me. 
I have accepted the updated invite and am looking forward to it. 


On Mon, Mar 31, 2025 at 3:01 PM Engage @easeverticalai <Engage@easeverticalai.com> wrote:
Hello Swagath,
We appreciate your response.

Would you be ok starting the call @ 4:30pm PST (15 min later) on the same day?


 Thanks again for your interest in our venture!

The Hiring Team @ Ease Vertical AI!
www.easeverticalai.com


On Mon, Mar 31, 2025 at 2:03 PM SWAGATH BABU <swagathb18@g.ucla.edu> wrote:
Hi Team,

Thanks so much for getting in touch.
I’m really excited about the opportunity to interview for the Applied ML Engineer role and to learn more about what you’re building at Ease Vertical AI.
I’ve scheduled the interview for April 2nd between 4:15–5:15 PM PST. Looking forward to our conversation.

On Mon, Mar 31, 2025 at 1:15 PM Engage @easeverticalai <Engage@easeverticalai.com> wrote:
Hello Swagath, 
We appreciate your interest in the above opportunity and would like to learn more about you!

Kindly use the link below to confirm some information and schedule a 1-hr introductory discussion with our leadership:
https://calendly.com/engage-easeverticalai/intro-founding-engineer-opportunity-ease-vertical-ai

 Thanks again for your interest in our venture!

The Hiring Team @ Ease Vertical AI!
www.easeverticalai.com


--
Thanks,
Warm regards
Swagath B
                                                                                

Swagath Babu

swagathb18@ucla.edu  
Linkedin: https://www.linkedin.com/in/swagathb/



--
Thanks,
Warm regards
Swagath B
                                                                                

Swagath Babu

swagathb18@ucla.edu  
Linkedin: https://www.linkedin.com/in/swagathb/



--
Thanks,
Warm regards
Swagath B
                                                                                

Swagath Babu

swagathb18@ucla.edu  
Linkedin: https://www.linkedin.com/in/swagathb/



--
Thanks,
Warm regards
Swagath B
                                                                                

Swagath Babu

swagathb18@ucla.edu  
Linkedin: https://www.linkedin.com/in/swagathb/



--
Thanks,
Warm regards
Swagath B
                                                                                

Swagath Babu

swagathb18@ucla.edu  
Linkedin: https://www.linkedin.com/in/swagathb/"""

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
