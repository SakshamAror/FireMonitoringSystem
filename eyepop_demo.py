# flake8: noqa
import os
import subprocess
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent
import json
from dotenv import load_dotenv
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt

load_dotenv()

api_key = os.getenv("EYEPOP_API_KEY")
print(f"Using API key: {api_key}")

threshold = 0.7  # Threshold for criticality index to trigger high-risk script

test_images = glob.glob('./test/*')
if not test_images:
    raise FileNotFoundError("No images found in the ./test folder.")
example_image_path = random.choice(test_images)

objectOfInterest = 'Person'
questionList = (
    "Is there a person in the image (Yes/No), "
    "How much danger is the person in if they are in the fire (as a decimal (2-3 decimal places) from 0-1 based on the criticality of the danger. 0 if no fire or no person)"
    "Report the values of the categories as classLabels. "
)

# Routes to YOLO model
def route_to_different_model(image_path):
    print("EyePop failed, running local model instead...")
    # Run Yolo model here, inputting image_path

# Function to call another Python script for high-risk images
def call_high_risk_script(image_path):
    print("High Danger found, triggering local YOLOv8 model for real-time monitoring...")
    # Run Yolo model here

with EyePopSdk.workerEndpoint(
    api_key=api_key
) as endpoint:
    prompt = f"Analyze the image of {objectOfInterest} provided and determine the categories of: " + questionList + "If you are unable to provide a category with a value then set its classLabel to null"

    print (f"Using prompt: {prompt}")

    endpoint.set_pop(
       Pop(components=[
            InferenceComponent(
                id=1,
                ability='eyepop.image-contents:latest',
                params={"prompts": [
                            {
                                "prompt": prompt
                            }
                        ] }
            )
        ])
    )

    try:
        result = endpoint.upload(example_image_path).predict()
        print(json.dumps(result, indent=4))
        if "classes" in result and len(result["classes"]) > 1:
            class_label = result["classes"][1]['classLabel']
            print("Criticality Index: " + str(class_label))

            try:
                index_value = float(class_label)
                is_high_danger = index_value >= threshold
            except ValueError:
                index_value = None
                is_high_danger = False
                print(f"Criticality index is not a number: {class_label}")

            print(f"Selected test image: {example_image_path}")
            img = Image.open(example_image_path)
            plt.imshow(img)
            plt.axis('off')

            if index_value is None:
                danger_text = "INVALID SCORE"
            elif index_value <= 0.5:
                danger_text = "LOW DANGER"
            elif index_value < threshold:
                danger_text = "MODERATE DANGER"
            else:
                danger_text = "HIGH DANGER"

            subtitle = (
                f"Criticality Index: {index_value:.3f}\nStatus: {danger_text}"
                if index_value is not None
                else "Criticality Index: Invalid"
            )

            plt.title("Selected Test Image")
            plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=11)
            plt.show()

            if is_high_danger:
                call_high_risk_script(example_image_path)
        else:
            print("Class label not found, routing to different script.")
            route_to_different_model(example_image_path)
    except Exception as e:
        error_str = str(e).lower()
        if 'timeout' in error_str or 'overload' in error_str or 'server' in error_str or 'healthy' in error_str:
            print(f"Timeout or server overload error: {e}. Routing to different script.")
            route_to_different_model(example_image_path)
        else:
            print(f"Other error: {e}")
