#%%
import base64
from openai import OpenAI
import re
client = OpenAI()
import json
import pandas as pd

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
#image_path = r"C:\Users\panka\OneDrive\Desktop\Research Papers for EB1A\Papers for Oil and Gas\OnGNet Paper_Vision\OGNetDevelopmentData\images\image_25.92380876474601_-97.48351579634272_0.png"

prompt ="""
You are a highly advanced AI model trained for aerial image classification tasks, specifically to detect the presence of oil and gas facilities.

---
### Task
Determine if a given aerial image contains an oil and gas facility. Return your conclusion in JSON format, including:
1. A binary indicator (FacilityPresent) where:
   - 1 indicates an oil and gas facility is present.
   - 0 indicates no oil and gas facility is present.
2. A detailed explanation (Reason) describing which visual cues influenced your classification.

---
### Step-by-Step Analysis

1. Identify Core Oil and Gas Features
   - Storage Tanks/Tank Farms: Look for circular or cylindrical shapes, often with large footprints and sometimes with visible shadowing or top covers.
   - Distillation or Processing Units: Complex, tower-like structures or clusters of equipment typically seen in refineries.
   - Pipelines and Manifolds: Linear structures connecting tanks or processing units.
   - Jetties/Piers for Coastal Facilities: Offshore docks extending into bodies of water, often accompanied by large onshore tanks or terminal facilities.

2. Check Surrounding Context
   - Industrial Footprint: Large open areas with consistent industrial patterns (roads, shipping containers, parking lots for trucks, etc.).
   - Presence of Power or Utility Infrastructure: High-voltage lines, flare stacks, or gas flare columns can suggest petrochemical operations.
   - Nearby Infrastructure: Roads, rail lines, or waterways specifically designed for heavy transport, potentially indicating large-scale industrial operations.

3. Rule Out Common Confusions
   - Chemical Plants: May also have tanks but often include clarifiers or small processing vessels that differ in layout from oil refineries.
   - Wastewater Treatment Plants: Look for sedimentation basins, aeration tanks, or clarifiers—these may appear as uniform circular or rectangular basins arranged in a grid.
   - Agricultural Silos: Can be mistaken for tanks; however, silos often appear in smaller clusters and are typically located in farmland settings rather than in large industrial complexes.

4. Assess Visibility and Confidence
   - Consider the clarity of the image (resolution, angle, lighting).
   - If features are ambiguous, rely on multiple indicators (e.g., combination of tanks, pipeline corridors, and visible processing units).

5. Formulate the Output
   - Make a definitive classification:
     - 1 if oil and gas facility features are clearly observed.
     - 0 if the scene lacks these features or contains only non-oil-and-gas structures.
   - Provide a concise but thorough explanation citing relevant visual evidence (e.g., “presence of cylindrical tank farms,” “complex distillation towers,” etc.).

---
### Output Format
Your output must be valid JSON with the keys FacilityPresent and Reason. Here are two example responses:

#### Example 1: Oil and Gas Facility Detected
{
  "FacilityPresent": 1,
  "Reason": "Multiple large cylindrical tanks and associated pipeline infrastructure are visible, indicating an oil and gas storage facility. The industrial layout with flare stacks further supports this classification."
}

#### Example 2: Not an Oil and Gas Facility
{
  "FacilityPresent": 0,
  "Reason": "The area shows a series of round clarifiers and rectangular basins typical of a wastewater treatment plant. No refining or large-scale tank structures are evident."
}
"""

# Function to process an image and get a response from the model
def classify_image(image_path):
    # Encode image to base64
    base64_image = encode_image(image_path)
    
    # Get the response from the OpenAI model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )
    
    # Extract content from the response
    response_content = response.choices[0].message.content.strip()
    return response_content

# Function to convert the model's response into a JSON-compatible format
import json

def convert_to_json(response):
    # Clean the response to extract JSON content
    cleaned_response = response.strip("```json").strip("```").strip()
    
    # Parse the cleaned response as JSON
    try:
        data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        data = {"FacilityPresent": None, "Reason": "Unable to parse response"}
    
    return data


#%% List of image paths (you can replace this with the actual list of images you want to classify)
# image_paths = [
#     r"images/image_47.99194648691705_-102.629177677087_0.png",
#     r"images/image_30.774497911656095_-89.86675877587317_0.png",
#     r"images/image_35.389217494922484_-119.04379135581819_1.png"
# ]
image_paths=pd.read_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\test.csv")
image_paths=image_paths["Image_Path"].to_list()
# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Image_Path", "FacilityPresent", "Reason"])

# Process each image and populate the DataFrame
for image_path in image_paths:
    response = classify_image("C:/Users/panka/OneDrive/Documents/GitHub/llm-oil-gas-vision/Data/"+image_path)
    result_json = convert_to_json(response)
    
    # Create a temporary DataFrame for the new row
    new_row = pd.DataFrame([{
        "Image_Path": image_path,
        "FacilityPresent": result_json["FacilityPresent"],
        "Reason": result_json["Reason"]
    }])

    # Use pd.concat to add the new row
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    # Print the DataFrame
    print(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\results\GPT\GPT_test_prediction.csv", index=False)

results_df

#%%