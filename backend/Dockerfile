FROM python:3.10-slim

WORKDIR /app

# Install dos2unix to handle line ending conversion from Windows
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Convert line endings and make entrypoint script executable
# This handles CRLF to LF conversion for files created on Windows
RUN dos2unix entrypoint.sh && chmod 755 entrypoint.sh
RUN dos2unix init_db.py

# Convert line endings for any additional shell scripts and Python files
# This ensures all executable files work properly regardless of development OS
RUN find . -name "*.sh" -exec dos2unix {} \; || true
RUN find . -name "*.py" -exec dos2unix {} \; || true

EXPOSE 5000

# Use sh to execute the script
CMD ["sh", "./entrypoint.sh"]