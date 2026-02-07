# Deployment Guide: Chess Puzzle Generator

This project is a FastAPI application that serves a web interface for generating chess puzzles. It can be deployed to any web host that supports Docker or Python applications.

## 1. Local Deployment (Docker)

To verify the project locally using Docker:

1. **Build the image**:
   ```bash
   docker build -t chess-puzzle-gen .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 chess-puzzle-gen
   ```

3. **Access the app**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 2. Cloud Deployment (Hosted Platforms)

### Option A: Render (Recommended)
1. Create a new **Web Service** on [Render](https://render.com/).
2. Connect your GitHub repository.
3. Select **Docker** as the Runtime.
4. Render will automatically use the `Dockerfile` to build and deploy your app.
5. In the "Environment" section, you may need to increase the memory if `torch` requires it (at least 1GB recommended).

### Option B: Railway
1. Create a new Project on [Railway](https://railway.app/).
2. Connect your GitHub repository.
3. Railway will detect the `Dockerfile` and deploy it automatically.

---

## 3. Server Deployment (VPS / Linux)

If you have your own server (DigitalOcean, AWS, etc.):

1. **Install Docker** on your server.
2. **Transfer your files** to the server.
3. **Run the Docker container** in the background:
   ```bash
   docker build -t chess-puzzle-gen .
   docker run -d -p 80:8000 --name chess-app --restart always chess-puzzle-gen
   ```
   *Note: This maps port 8000 of the container to port 80 of your server, making it accessible via your server's IP or domain.*

---

## Important Notes
- **Models**: Ensure `fen_generator.pth` and other model files are pushed to your repository or uploaded to the server. These are the weights for the Transformer generator.
- **Resources**: Machine learning models can be memory-intensive. For production, at least 2GB of RAM is highly recommended.
- **Port Mapping**: The FastAPI app defaults to port 8000. Ensure your firewall allows traffic on this port or map it accordingly in Docker.

## 🔒 Best Practices
- **Security**: For public deployment, use a reverse proxy like Nginx with SSL (Let's Encrypt).
- **Process Management**: In a non-Docker environment, use `gunicorn` with the `uvicorn` worker for better process management:
  ```bash
  pip install gunicorn
  gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
  ```
