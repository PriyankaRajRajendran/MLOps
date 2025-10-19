# Stage 1: Model Training
FROM python:3.9 AS model_train

WORKDIR /app

# Copy the source code for model training
COPY src/model_train.py /app/

# Install any needed packages specified in requirements.txt
COPY Requirements.txt /app/
RUN pip install --no-cache-dir -r Requirements.txt

# Run the model training script
RUN python model_train.py

# Stage 2: Serve Predictions
FROM python:3.9 AS serving

WORKDIR /app

# Copy the trained model and preprocessing objects from the previous stage
COPY --from=model_train /app/wine_model.keras /app/
COPY --from=model_train /app/scaler.pkl /app/
COPY --from=model_train /app/wine_info.pkl /app/

# Copy the source code for serving predictions
COPY src/main.py /app/

# Copy the templates directory
COPY src/templates /app/templates

# Copy the statics directory
COPY src/static /app/statics

# Install any needed packages specified in requirements.txt
COPY Requirements.txt /app/
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r Requirements.txt

# Expose port 4000
EXPOSE 4000

# Define environmental variables
ENV NAME World

# Run the Flask application for serving predictions
CMD ["python", "main.py"]