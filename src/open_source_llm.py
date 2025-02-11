#%%
import ollama
image_path = r"C:\Users\panka\OneDrive\Desktop\Research Papers for EB1A\Papers for Oil and Gas\OnGNet Paper_Vision\OGNetDevelopmentData\images\image_25.92380876474601_-97.48351579634272_0.png"
prompt ='''
You are a highly advanced AI model trained for image classification tasks and reasoning based on aerial imagery. Your task is to classify whether a given aerial image contains an oil and gas facility or not. You must provide the classification and the reasoning behind your decision in JSON format.

Steps to Determine:
Analyze the image for features characteristic of oil and gas facilities, such as:
Storage tanks and tank farms: Circular or cylindrical shapes, often with large footprints.
Distillation units: Complex structures unique to oil refineries.
Jetties and piers: Common for coastal refineries and liquefied natural gas (LNG) terminals.
Consider any potential false positives from similar-looking facilities:
Chemical plants: Identified by clarifiers or small processing tanks.
Grain processing facilities: Recognized by grain bins and storage warehouses.
Wastewater treatment facilities: Identified by sedimentation tanks and clarifiers.
Leverage contextual clues like surrounding landscapes or associated features (e.g., industrial footprints, pipelines, nearby infrastructure).
Output Format:
Your output must be in JSON format with the following keys:

FacilityPresent: A binary classification where 1 indicates the presence of an oil and gas facility and 0 indicates its absence.
Reason: A detailed explanation of why the image was classified as such. Include features identified in the image that influenced the decision.

Example JSON Output:
Case 1: Oil and Gas Facility Detected
{
  "FacilityPresent": 1,
  "Reason": "The image contains large round storage tanks and distillation units, which are characteristic of oil refineries. Additionally, the industrial layout with pipelines and associated infrastructure supports this classification."
}

Case 2: Not an Oil and Gas Facility
{
  "FacilityPresent": 0,
  "Reason": "The image contains grain bins and storage warehouses, which are typical of a grain processing facility, not an oil and gas facility. No storage tanks or distillation units were detected."
}

'''
#%%
response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': prompt,
        'images': [image_path]
    }]
)

print(response)
#%%