# MedXpert Docker Guide

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

## Getting Started

### For Team Members

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd medxpert
   ```

2. Start the application:
   ```bash
   docker-compose up -d
   ```

3. Access the application at:
   ```
   http://localhost:5000
   ```

4. Stop the application:
   ```bash
   docker-compose down
   ```

### For Development

1. Make changes to the code
2. Rebuild and restart the container:
   ```bash
   docker-compose up -d --build
   ```

3. View logs:
   ```bash
   docker-compose logs -f
   ```

## Data Persistence

- The `static` directory (including uploads) is mounted as a volume, so data will persist between container restarts
- The models directory is also mounted as a volume to avoid rebuilding the image when models change

## Troubleshooting

1. If you encounter permission issues with the mounted volumes:
   ```bash
   sudo chown -R $USER:$USER static models
   ```

2. If the application fails to start, check the logs:
   ```bash
   docker-compose logs
   ```

3. To access the container shell:
   ```bash
   docker-compose exec medxpert bash
   ```

4. If the models are too large for building the image, you can:
   - Add the models to .dockerignore
   - Mount the models directory as a volume (already configured)
   - Download the models during container initialization (requires modifying the Dockerfile)

## Updating the Application

1. Pull the latest changes:
   ```bash
   git pull
   ```

2. Rebuild and restart:
   ```bash
   docker-compose up -d --build
   ``` 