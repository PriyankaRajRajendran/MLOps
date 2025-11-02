# Docker Lab 3 - Wine Quality Classification API

A machine learning web application that classifies wine types based on chemical properties, built with TensorFlow, Flask, and Docker using multi-stage builds.

## About the Project

This project implements a neural network classifier to predict wine class (Class 0, Class 1, or Class 2) based on 13 chemical measurements from the UCI Wine dataset:
- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

The application uses a **multi-stage Docker build** to:
1. Train the model in the first stage
2. Serve predictions via Flask API in the second stage

##  Features

- **Multi-stage Docker build** for efficient containerization
- **Neural network model** with 97%+ accuracy
- **Interactive web interface** with 13 input fields
- **RESTful API** for programmatic access
- **Real-time predictions** with confidence scores
- **Visual probability bars** for all wine classes

##  Technologies Used

- **Python 3.9**
- **TensorFlow 2.15** 
- **Flask** 
- **scikit-learn** 
- **Docker** 

##  Project Structure
```
docker_lab3/
├── src/
│   ├── model_training.py       # Neural network training script
│   ├── main.py                 # Flask API server
│   ├── templates/
│   │   └── predict.html        # Web interface
│   └── statics/                # Screenshots and images
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt            # Python dependencies
└── DockerRunCommands                       # Build and run commands
```

##  How to Run

### Prerequisites
- Docker Desktop installed and running
- Signed in to Docker Hub
- Verify through mail

### Build the Docker Image
```bash
docker build -t wine-classifier .
```

This command will:
- Download Python 3.9 base image
- Install all dependencies
- Train the neural network model (~2 minutes)
- Create the serving container

### Run the Container
```bash
docker run -p 4000:4000 wine-classifier
```

### Access the Application

Open your browser and go to:
```
http://localhost:4000/predict
```

##  Example Test Cases

### Wine Class 0 Sample
- Alcohol: 13.2%
- Malic Acid: 1.78 g/L
- Ash: 2.14 g/L
- Alcalinity: 11.2
- Magnesium: 100 mg/L
- Total Phenols: 2.65 g/L
- Flavanoids: 2.76 g/L
- Nonflavanoid Phenols: 0.26 g/L
- Proanthocyanins: 1.28 g/L
- Color Intensity: 4.38
- Hue: 1.05
- OD280: 3.4
- Proline: 1050 mg/L

### Wine Class 1 Sample
- Alcohol: 12.5%
- Malic Acid: 2.5 g/L
- Ash: 2.4 g/L
- Alcalinity: 19.0
- Magnesium: 85 mg/L
- Total Phenols: 1.8 g/L
- Flavanoids: 1.5 g/L
- Nonflavanoid Phenols: 0.4 g/L
- Proanthocyanins: 1.0 g/L
- Color Intensity: 5.5
- Hue: 0.8
- OD280: 2.5
- Proline: 450 mg/L

### Wine Class 2 Sample
- Alcohol: 13.0%
- Malic Acid: 3.2 g/L
- Ash: 2.6 g/L
- Alcalinity: 21.0
- Magnesium: 95 mg/L
- Total Phenols: 1.5 g/L
- Flavanoids: 0.8 g/L
- Nonflavanoid Phenols: 0.45 g/L
- Proanthocyanins: 0.9 g/L
- Color Intensity: 7.0
- Hue: 0.6
- OD280: 1.8
- Proline: 550 mg/L

##  Model Performance

- **Architecture**: 4-layer Neural Network
- **Input Features**: 13 chemical properties
- **Output Classes**: 3 wine types
- **Training Epochs**: 200
- **Test Accuracy**: ~97-99%
- **Dataset**: UCI Wine Dataset (178 samples)

##  Docker Multi-Stage Build

The Dockerfile uses two stages:

**Stage 1 (Training):**
- Trains the model using `model_training.py`
- Saves `wine_model.keras`, `scaler.pkl`, and `wine_info.pkl`

**Stage 2 (Serving):**
- Copies trained model from Stage 1
- Sets up Flask web server
- Exposes port 4000
- Runs the prediction API

## API Endpoints

### `GET /`
Returns welcome message

### `GET /predict`
Returns the web interface with input form

### `POST /predict`
Accepts form data with 13 wine features and returns:
```json
{
  "wine_class": "class_0",
  "confidence": 0.98,
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.015,
    "class_2": 0.005
  }
}
```

##  Stopping the Container

Press `Ctrl + C` in the terminal where the container is running.

##  Author

**Priyanka Raj Rajendran**  
Graduate Student - MS in Data Analytics Engineering  
Northeastern University

##  Course

MLOps - Neural Networks/Deep Learning  
Docker Labs - Lab 3

##  Dataset Information

The Wine dataset is a classic multi-class classification dataset from the UCI Machine Learning Repository. It contains chemical analysis results of wines grown in the same region in Italy but derived from three different cultivars.

##  Web Interface Features

- Clean, modern design with gradient background
- Two-column grid layout for better organization
- Example values provided for quick testing
- Real-time prediction with loading indicator
- Visual probability bars for each wine class
- Responsive design for mobile and desktop
