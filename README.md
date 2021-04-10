# Google Cloud Vision API - Use Case for detecting Facial Emotions within an Image

## Objective
To illustrate the steps and code required to use the Google Cloud Vision API for emotion detection within an image.

**Sample Use Case:**
A picture of friends
Picture Courtesy: [Westend61](https://www.westend61.de/en/imageView/AFVF05511/portrait-of-happy-friends-sitting-on-a-wall-outdoors)

![Happy Friends](https://user-images.githubusercontent.com/36125669/114277320-467a7a80-9a5d-11eb-95f4-9ba4266fbac1.jpg)


**Objective:**
Identify the emotion likelihood for all faces detected in the image.

### Prerequisites
Steps 1 and 2 are prerequisites to sign-up for the Google Cloud Platform and enable the Vision API.

#### Step 1 - Sign-up for the Google Cloud Platform
The [Google Cloud Platofarm](https://cloud.google.com/free/?utm_source=google&utm_medium=cpc&utm_campaign=japac-HK-all-en-dr-bkws-all-all-trial-e-dr-1009882&utm_content=text-ad-none-none-DEV_c-CRE_255875986060-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20GCP%20~%20Trial_cloud%20-%20create%20account-KWID_43700007271914961-kwd-58031179117&userloc_9069537-network_g&utm_term=KW_create%20a%20google%20cloud%20account&gclid=EAIaIQobChMIi7SosOHy7wIV7dVMAh0OeQfNEAAYASAAEgLP_fD_BwE&gclsrc=aw.ds) has a simple sign-up process that allows access to the free-tier for many of their key products - including the Vision API.

<img width="1440" alt="GCP Sign-Up" src="https://user-images.githubusercontent.com/36125669/114258161-42b40d00-99f7-11eb-98ab-e2ee2623ef85.png">

![GCP Services](https://user-images.githubusercontent.com/36125669/114258231-b6561a00-99f7-11eb-8d4a-4fb010eaf9b8.png)

#### Step 2 - Enable the Google Cloud Vision API

Google has a step by step set up [documentation](https://cloud.google.com/vision/docs/before-you-begin) that may be used to enable the Vision API via a new project that will grant a user access keys via a downloadable json file to be used in the code that follows.

A more detailed step by step process can be found [here](https://daminion.net/docs/topics/auto-tagging/how-to-get-google-cloud-vision-api-key/)

The primary output from this step will be the JSON file with your credentials to access the Vision API during the project.

****Note on Cost:**** The first 1000 units are free with a small incremental charge post. Details [here](![Vision API Cost](https://user-images.githubusercontent.com/36125669/114277686-ce14b900-9a5e-11eb-8e2e-91cfbb886cd1.jpeg)


#### Step 3 - Import all the important libraries for this project

```
import cv2
import imutils
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd
import requests
import time
from base64 import b64encode
from IPython.display import Image
from google.cloud import vision
from google.cloud.vision import AnnotateFileRequest
from google.protobuf.json_format import MessageToDict
```

#### Step 4 - Import GCP credentials from the JSON document downloaded

Please provide the path where your JSON document is located /drive

```
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="GCP - Vision API JSON.json"
```

#### Step 5 - Import the image with the face/s on which emotion detection is to be run

```
import io

path = 'Happy Friends.jpg'
with io.open(path, 'rb') as image_file:
        content = image_file.read()
```

#### Step 6 - # Map Vision API

```
image = vision.Image(content=content)
```
```
client = vision.ImageAnnotatorClient()
```

#### Step 7 - Call Vision API

```
response = client.face_detection(image=image)
faces = response.face_annotations
```

## Output: Facial Emotion Detection Likelihood

```
likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                   'LIKELY', 'VERY_LIKELY')
print('Faces:')

for face in faces:
    print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
    print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
    print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
    print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in face.bounding_poly.vertices])

    print('face bounds: {}'.format(','.join(vertices)))

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
```

Faces:
joy: VERY_LIKELY
anger: VERY_UNLIKELY
sorrow: VERY_UNLIKELY
surprise: VERY_UNLIKELY
face bounds: (262,182),(417,182),(417,362),(262,362)
joy: VERY_UNLIKELY
anger: VERY_UNLIKELY
sorrow: VERY_UNLIKELY
surprise: VERY_UNLIKELY
face bounds: (770,179),(933,179),(933,368),(770,368)
joy: VERY_LIKELY
anger: VERY_UNLIKELY
sorrow: VERY_UNLIKELY
surprise: VERY_UNLIKELY
face bounds: (402,216),(549,216),(549,387),(402,387)
joy: VERY_LIKELY
anger: VERY_UNLIKELY
sorrow: VERY_UNLIKELY
surprise: VERY_UNLIKELY
face bounds: (582,239),(710,239),(710,388),(582,388)

## Author
Neil Shastry
