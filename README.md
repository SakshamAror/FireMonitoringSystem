# FireMonitoringSystem

Our AI-driven multi-model fire risk detection system combines two complementary computer vision models to identify growing fires and detect people at risk in a timely manner, enabling first responders to intervene before harm occurs.

The system will operate in two stages. First, a custom EyePop.ai cloud-hosted model will analyze images captured from video streams (e.g., CCTV feeds) at five-minute intervals. This model computes a criticality index on a scale from 0 to 1, representing the likelihood that a person is in danger due to fire or smoke. If the criticality index reaches 0.7 or higher, the scene is flagged as High Danger.

Once a stream is marked High Danger, the system will activate a second model: a YOLOv8 pre-trained object detection model enhanced with HSV color analysis. This model performs real-time, localized fire and smoke analysis, providing visual risk indicators that help first responders assess severity and act accordingly on site.

Because the EyePop model is cloud-based and may be subject to connectivity or availability issues, this repository also includes a local YOLOv8-only fallback pipeline. This backup model can analyze single images offline and output a visual fire-risk assessment, ensuring continued functionality even when cloud access is unavailable.

The demo files in this GitHub repository simulate how the EyePop and YOLOv8 models would function in a real-world monitoring system. Example use cases include deployment in forests, schools, universities, large buildings, and other high-risk environments where early fire detection and human safety are critical.

Datasets used:
- Smoke and fire detection dataset:
  - https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data
  - https://www.kaggle.com/datasets/phylake1337/fire-dataset/data?select=fire_dataset
- People detection dataset:
  - https://www.kaggle.com/datasets/adilshamim8/people-detection
