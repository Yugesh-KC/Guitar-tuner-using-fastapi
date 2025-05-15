# FastAPI Guitar Tuner

A simple real-time guitar tuner using FastAPI and WebSockets. It listens to your microphone, detects pitch, and helps you tune your guitar—all in your browser.

---

## How to Run

### Locally

1.  **Clone and enter the repository:**

    ```bash
    git clone https://github.com/Yugesh-KC/Guitar-tuner-using-fastapi
    cd your-repo
    ```
2.  **Set up a virtual environment and install dependencies:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Start the server:**

    ```bash
    uvicorn app.main:app --reload
    ```
4.  **Open your browser:**

    Open your browser at http://localhost:8000
    
### With Docker

1.  **Build the Docker container:**
    ```bash
    docker build -t fastapi-tuner .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 fastapi-tuner
    ```

3.  **Open your browser:**
    Navigate to [http://localhost:8000](http://localhost:8000)