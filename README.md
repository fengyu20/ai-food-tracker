
# Eczema AI Food Tracker: Using YOLO, Local LLM, and Gemini

## Background

Many people suffer from eczema, and while environmental and dietary factors are known to be common triggers, these triggers are highly individual. What affects one person may not affect another.

It’s helpful to build **a personal trigger database** — so that when a flare-up happens, you can trace it back to potential causes more easily.

When it comes to dietary factors, manually recording every meal is time-consuming and often inconsistent. This project simplifies this process: You just take a photo of your food, and the system automatically detects and logs the food items. These entries become part of your personal eczema trigger history.

> Note: Currently, only the AI-based food recognition component has been implemented. Other parts are not yet included.

## Summary

The project's goal is that when given an image, it can extract the relevant food items inside the image.

The project started with a **YOLO-based** object detection model. While effective for basic object detection, YOLO required more setup and did not scale well for fine-grained food classification.

To improve accuracy and reduce configuration overhead, the project transitioned to using vision-capable Large Language Models (**LLMs**). These models offer broader generalization and require less custom training.

For those interested in the development process, the original YOLO implementation is still included in the project files.

## Dataset

This project uses the [Food Recognition 2022 dataset](https://datasetninja.com/food-recognition).

The dataset contains 43,962 images with 95,009 labeled objects across 498 food classes.

> Note: Although human-verified and checked, there are still mistakes in the datasets. For simplicity, we won’t address the label mistakes. 

## Model Details

### LLM: Large Language Models

After using YOLO to get rudimentary results, I realized that it would be hard to utilize without extensive training to get suitable weights. Also, going back to the dataset website itself, even the [prize winners](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022/leaderboards) didn’t achieve the desired precision rate.  

But when I tried uploading the food image to different LLMs, they generally returned accurate results. So this project switched to the LLM approach.  

#### File Structure
```
models/                          # LLM-based food image classification
├── classify.py                  # Classify single food image
├── evaluate.py                  # Batch evaluation with performance reports
│
├── core/                        # Core classification system
│   ├── classifier.py           # Main classification orchestrator
│   ├── taxonomy.py             # Food category matching and validation
│   ├── prompts.py              # Prompts
│   ├── provider.py             # Abstract provider interface
│   ├── validator.py            # Ground truth comparison
│   ├── common.py               # Shared utilities and helpers
│   └── settings.py             # Configuration paths and constants
│
├── providers/                   # LLM providers
│   ├── gemini.py               # Google Gemini API
│   ├── openrouter.py           # OpenRouter API  
│   └── llama.py                # Local Ollama/Llama models
│
├── utils/                       # Taxonomy building utilities
│   ├── common.py               # Normalization and CLI helpers
│   ├── populate_taxonomy.py    # Generate initial food taxonomy
│   └── refine_taxonomy.py      # Multi-model taxonomy refinement
│
└── configs/
    └── provider_config.json    # Model parameters and settings
```

#### Providers  
1. **Local LLM - Llama**: In the beginning, I tried to use a local LLM. However, the training also relies on a GPU machine, which I do not have. So for testing, I recommend only using the classify module to test a single image.  
2. **Single Provider - Gemini**: Google Gemini offers free-tier API use, therefore it is suitable for debug testing.  
3. **Multiple Providers - OpenRouter**: After discussing with my mentor, James, from WISJ Summer School, he also recommended using OpenRouter, which is convenient as it provides a central hub to call different LLM providers, charging a small service fee (8%).  

#### How to
Please first export your own gemini / openrouter key in your local env.
```
export OPENROUTER_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

Then run the following commands:
- Signle image testing: `python -m models.classify path/to/image.jpg --provider gemini`
- Batch evaluation: `python -m models.evaluate --providers gemini openrouter`
    - *Note*: This command will use models provided in [the config file](/models/configs/provider_config.json), update if needed.


### Object Detection: YOLO Training

YOLO stands for "You Only Look Once," which is a famous object detection model.

This project uses yolov11m as the base model, and you can find all the details in [this Jupyter notebook](/yolo_training.ipynb). 

It contains the following sections:
0. **Environment setup**
1. **Download the food image dataset**  
2. **Training preparation** 
3. **Training**

#### 0. Environment setup
Make sure you have the GPU activated (you can still train on CPU; however, it can be very slow). Install and import necessary packages.

#### 1. Download the food image dataset
Depending on your needs, you can download the full dataset (3GB, 40k+ images) or the sample dataset (300MB, 4k+ images).

#### 2. Training preparation

##### Filter categories that have few samples
The dataset has an imbalanced data distribution, as you can see from the dataset website. To better generalize YOLO's performance, this project only focuses on categories that have more than 30 images. So among 498 full food categories, only 410 food categories remain.

##### Convert annotation JSON to YOLO detection bounding box

For example, for `training/img/006497.jpg`, annotations can be found in the relevant annotation folder: `training/ann/006497.jpg.json`

However, YOLO needs a label file in text format.

You can use [the following image](https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif) to better understand how it works:

![yolo sample img](https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif)

The corresponding label text is as follws:

```txt
0 0.481719 0.634028 0.690625 0.713278
0 0.741094 0.524306 0.314750 0.933389
27 0.364844 0.795833 0.078125 0.400000
```

- The first column (`0`) is the category ID. In this case, `0` refers to `Person`.
- The second (`0.481719`) and third columns (`0.634028`) are used as x,y central point coordinates of this object.
- The fourth(`0.690625`) and fifth columns(`0.713278`) represent the width and height of the object.


For more information, you can check [this documentation](link).

**Note:** In our project, the original ID in `meta.json` is not used. The `class_map` returns a unique integer ID as the key.

##### Prepare the dataset.yaml

For YOLO training, it’s important to let YOLO know your training targets. In our case, these are food categories that can be found in our image datasets, as well as the paths of our training/validation dataset.

> Note: The originally downloaded dataset includes the image path as “training/img”; however, the YOLO convention expects “training/images,” so the rename is needed.

#### 3. YOLO Training
This part selects the suitable training model, parameters, and starts training. The best model will be saved as best.pt.


## Evaluation

A challenge faced is that YOLO and LLMs produce totally different outputs: YOLO produces a list of `[bounding_box, class_id, confidence_score]`, while LLMs are more open-ended — answers can change a lot depending on the prompt.

### Goal: Extract food items from the image
This project tries to find a common ground to compare the results by focusing on identifying the food items present in an image.

Specifically, YOLO’s final list would be the food categories whose confidence scores pass a certain threshold, and LLMs would follow a specific prompt to return the corresponding food categories in JSON.

### Metrics
- **Precision**: Of the items the model identified, how many were correct?
- **Recall**: Of all the items that were actually in the image, how many did the model find?
- **F1-Score**: The mean of Precision and Recall, giving a single score for recognition accuracy.
