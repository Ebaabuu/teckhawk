# TechHawk IT Help Desk 
A local Retrieval-Augmented Generation (RAG) prototype built for the Edwards Campus IT department.

## How to Run Locally

1. **Install Dependencies:**
	Open your terminal and run:
	`pip install -r requirements.txt`

2. **Add Your API Key:**
	Create a file named `.env` in the root directory. Inside, add your API key:
	`GOOGLE_API_KEY=your_key_here`

3. **Build the Database (Phase A):**
	Add your markdown manuals to the `docs/` folder, then run the ingestion script to build the ChromaDB vector store:
	`python ingest.py`

4. **Launch the Web App (Phase B):**
	Start the Streamlit interface:
	`python -m streamlit run app.py`
	
Delete Later: To add to github, run the following in terminal:
1. git add [Name of file changed]
2. git commit -m "Message to specify changes"
3. git push