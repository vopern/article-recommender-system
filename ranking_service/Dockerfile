# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /ranking_service

# Copy the requirements file
COPY ./requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY ./ranking_service .
COPY ./utils.py /
COPY ./config.yaml /
# Expose the port that the api will run on
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Set the command to run the Streamlit script
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]