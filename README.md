# Intelligent Traffic System for Urban Conditions

A real-time intelligent traffic control system using video-based vehicle tracking, violation detection, and dynamic traffic management.

## Features

- Real-time vehicle detection and tracking using YOLOv5 and AlexNet V3
- Dynamic traffic signal control based on traffic density
- Traffic violation detection (helmet, seatbelt, lane violations, speeding)
- Number plate recognition using OCR
- Emergency vehicle priority system
- Real-time analytics and visualization dashboard
- Data logging and reporting capabilities

## System Requirements

### Hardware
- Raspberry Pi 4 (4GB RAM variant)
- Camera module (1080p resolution minimum)
- Power supply and appropriate casing

### Software
- Python 3.8+
- Docker
- AWS CLI (for deployment)
- Git

## Project Structure

```
intelligent_traffic_system/
├── src/
│   ├── detection/           # Vehicle detection and tracking modules
│   ├── violation/           # Traffic violation detection logic
│   ├── signal_control/      # Traffic signal control system
│   ├── ocr/                 # Number plate recognition
│   ├── emergency/           # Emergency vehicle handling
│   ├── analytics/           # Data analysis and reporting
│   └── utils/              # Helper functions and utilities
├── ui/                     # Streamlit web interface
├── models/                 # Pre-trained models and weights
├── tests/                  # Unit and integration tests
├── config/                 # Configuration files
├── data/                   # Data storage and logs
├── docs/                   # Documentation
└── docker/                 # Docker configuration files
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intelligent-traffic-system.git
   cd intelligent-traffic-system
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Download pre-trained models:
   ```bash
   python scripts/download_models.py
   ```

## Running the System

### Development Mode
```bash
python src/main.py
```

### Using Docker
```bash
docker-compose up --build
```

### Running Tests
```bash
pytest tests/
```

## Deployment

### Local Deployment (Raspberry Pi)
1. Install Raspbian OS
2. Clone the repository
3. Follow setup instructions above
4. Configure system service for automatic startup

### AWS Deployment
1. Configure AWS credentials
2. Run deployment script:
   ```bash
   python scripts/deploy_aws.py
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 team for object detection
- OpenCV community
- Streamlit team for the UI framework 