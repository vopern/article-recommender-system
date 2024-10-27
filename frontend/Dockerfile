# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /frontend

# Copy the requirements file
COPY ./requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit script
COPY ./frontend .

# Expose the port that Streamlit will run on
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Set the command to run the Streamlit script
CMD ["streamlit", "run", "frontend.py"]