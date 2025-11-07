EMAIL Agent - RAG prototype

# 📧 Email Digital Twin – Initial Prototype

This project is an **early prototype** of a larger **Email Digital Twin system** — an agentic AI workflow that automatically reads, understands, and drafts context-aware email responses grounded in attached documents (like PDFs).  

It combines **LangChain**, **FAISS**, and **OpenAI GPT models** to create a retrieval-augmented email responder that mimics Swagath’s communication style and tone.

---

## 🚀 Features

- **Email Thread Parsing:** Extracts the latest email from a conversation chain.  
- **PDF Context Retrieval:** Reads and chunks PDF attachments for reference.  
- **FAISS Vector Search:** Embeds and indexes document chunks for semantic retrieval.  
- **GPT-Powered Reply Generation:** Generates concise, polite, and context-aware responses using OpenAI’s GPT model.  
- **Persona Customization:** Uses a `persona.json` file to personalize tone, phrases, and sign-offs.  
- **Response Logging:** Saves all generated replies and context for audit and iteration.

---

## 🧩 Project Structure

```bash
email_digital_twin_prototype/
│
├── main.py                   # Main pipeline to extract, retrieve, and generate replies
├── persona.json              # Defines tone, phrases, and signature for personalized emails
│
├── data/                     # Folder for storing input files
│   └── myfile.pdf            # Example PDF attachment used in email context
│
├── vectorstore/              # FAISS embeddings store (auto-created on runtime)
│
├── response_log.json         # Auto-generated log of all email responses
├── .env                      # Stores environment variables (e.g., OPENAI_API_KEY)
├── requirements.txt          # List of required Python dependencies
└── README.md                 # Project documentation
```

Just a prototype for bigger project
