FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Expose the port for the Streamlit app
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/INTERFAZ.py", "--server.port=8501", "--server.address=0.0.0.0"]