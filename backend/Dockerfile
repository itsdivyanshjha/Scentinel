FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make entrypoint script executable with more explicit permissions
RUN chmod 755 entrypoint.sh

EXPOSE 5000

# Use sh to execute the script instead of relying on execute permissions
CMD ["sh", "./entrypoint.sh"]