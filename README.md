# Article Research Tool

This project is a Streamlit app that:

- accepts one or more article URLs
- extracts and chunks article content
- creates embeddings with `sentence-transformers`
- stores vectors in a local FAISS index
- answers questions over the processed articles using Groq

## Runtime

This app is pinned to Python 3.11 through `runtime.txt`. That avoids deployment issues with the PyTorch ecosystem on newer Python versions.

## Local setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

4. Start the app:

```bash
streamlit run main.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the app entrypoint to `main.py`.
4. In the app settings, add this secret:

```toml
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL="llama-3.3-70b-versatile"
```

5. Deploy the app.

## Important notes

- Do not commit your real `.env` file or API keys.
- `llama3-70b-8192` was retired by Groq on August 30, 2025. This app now defaults to `llama-3.3-70b-versatile`.
- Rotate the current Groq key if it has been exposed anywhere outside your machine.
- `faiss_store.pkl` is generated locally and may be recreated when the app restarts on hosted platforms.
- The FAISS file can be large, so avoiding committing generated indexes is usually better for deployment.
