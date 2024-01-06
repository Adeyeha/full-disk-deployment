FROM python:3.9

WORKDIR /app

# Copy just the requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the start script is executable
COPY start.sh .
RUN chmod +x start.sh

# Set the start script as the default command
CMD ["./start.sh"]
