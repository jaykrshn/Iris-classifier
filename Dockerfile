# Use the official Python image as a base image
FROM python:3.9.19

# Set the working directory in the container
WORKDIR /Iris-classifier

# Copy the application code to the working directory
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code to the working directory
COPY ./ /Iris-classifier/

CMD ["python", "train.py"]
