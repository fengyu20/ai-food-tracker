
# Eczema AI Food Tracker: Using YOLO and LLM

![overview](/file/overview.jpg)

## Table of Contents

- [Background](#background)
- [Summary](#summary)
- [Dataset](#dataset)
- [Model Details](#model-details)
  - [Object Detection: YOLO Training](#object-detection-yolo-training)
    - [0. Environment setup](#0-environment-setup)
    - [1. Download the food image dataset](#1-download-the-food-image-dataset)
    - [2. Training preparation](#2-training-preparation)
      - [Filter categories that have few samples](#filter-categories-that-have-few-samples)
      - [Convert annotation JSON to YOLO detection bounding box](#convert-annotation-json-to-yolo-detection-bounding-box)
      - [Prepare the dataset.yaml](#prepare-the-datasetyaml)
    - [3. YOLO Training](#3-yolo-training)
  - [LLM: Large Language Models](#llm-large-language-models)
    - [File Structure](#file-structure)
    - [Providers](#providers)
    - [How to](#how-to)
- [Evaluation](#evaluation)
  - [Challenges Faced](#challenges-faced)
  - [Solutions](#solutions)
  - [Metrics](#metrics)
  - [Sample Results](#sample-results)
- [Takeaways](#takeaways)

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

### Object Detection: YOLO Training

YOLO stands for "You Only Look Once," which is a famous object detection model.

This project uses yolov11m as the base model, and you can find all the details in [this Jupyter notebook](/yolo_training.ipynb). 

It contains the following sections:

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

**Note:** In our project, the original ID in `meta.json` is not used. The `class_map` returns a unique integer ID as the key.

##### Prepare the dataset.yaml

For YOLO training, it’s important to let YOLO know your training targets. In our case, these are food categories that can be found in our image datasets, as well as the paths of our training/validation dataset.

> Note: The originally downloaded dataset includes the image path as “training/img”; however, the YOLO convention expects “training/images,” so the rename is needed.

#### 3. YOLO Training
This part selects the suitable training model, parameters, and starts training. The best model will be saved as best.pt.

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
├── providers/                  # LLM providers
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
    ├── config.py               # Normalization and CLI helpers
    ├── evaluation_config.yml   # Config file
    └── provider_config.json    # Model parameters and settings
```

#### Providers  
1. **Local LLM - Llama**: In the beginning, I tried to use a local LLM. However, the training also relies on a GPU machine, which I do not have. So for testing, I recommend only using the classify module to test a single image.  
2. **Single Provider - Gemini**: Google Gemini offers free-tier API use, therefore it is suitable for debug testing.  
3. **Multiple Providers - OpenRouter**: After discussing with my mentor, James, from WISJ Summer School, he also recommended using OpenRouter, which is convenient as it provides a central hub to call different LLM providers, charging a small service fee (8%).  

#### How to
1. In the root folder, activate the virtual enviroment and install dependencies.
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Then, export your own gemini / openrouter key in your local env.
```
export OPENROUTER_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

3. (Optional) This repo contains a few images for testing, if you want to use the full (40k+) / sample (4k+) dataset, run the following command:
```
python download_dataset.py --type sample / full
```

4. After that,  run the following commands:
- Signle image testing: `python -m models.classify content/extracted_food_recognition/validation/img/008108.jpg --provider gemini`
- Batch evaluation: `python -m models.evaluate --providers gemini openrouter`
    - *Note*: This command will use models provided in [the config file](/models/configs/provider_config.json), update if needed.



## Evaluation

A challenge faced is that YOLO and LLMs produce totally different outputs: YOLO produces a list of `[bounding_box, class_id, confidence_score]`, while LLMs are more open-ended — answers can change a lot depending on the prompt.

This project tries to find a common ground to compare the results by focusing on identifying the food items present in an image, with a focus on comparing LLMs results.

### Challenges Faced 
1.**Label quality**: Although around 500 food categories are provided, they sometimes lack a principle (sometimes the label focuses on the compound dish name, sometimes on ingredients); sometimes labels are wrong (mistaking feta for tofu); and sometimes annotations include attributes that cannot be easily detected from the image (with/without salt).  
2. **LLM responses can be wrong**: Even though prompts are detailed and we expect the LLM to follow our rules (find the closest category), it can return food outside the category.  
3. **LLM responses are flexible**, but our labels are fixed. Sometimes LLMs return "dried figs," but the original label is "fig-dried," causing troubles in evaluation.

### Solutions
1. Expand ground truth’s rigid labels with similar items: 
    - In `populate_taxonomy.py`(/models/utils/populate_taxonomy.py), we first use an LLM to find the parent category and then some similar items. 
    - As it is used as a taxonomy, we then use `refine_taxonomy.py`(models/utils/refine_taxonomy.py) to have a second LLM challenge the result. When disagreement arises, we use a third LLM to synthesize the final JSON. 
    - Finally, we use a general normalization strategy, for example, lowercasing and removing punctuation, and removing common preparation methods like "cooked." In this case, we solve the "fig-dried" vs. "dried figs" mismatch.  
```json
"fig-dried": {
    "parent_category": "Raw Ingredient - Fruits",
    "similar_items": [
        "dried figs",
        "prunes",
        "raisins",
        "dried apricots",
        "dried dates"
    ],
```
2. When evaluating, if the previous step fails, it will trigger the next step until the end.
    - **Exact match:** The project compares the LLM’s predicted items against the expanded taxonomy (including `similar_items`).  
    - **Lemma Match**: It then checks if the core food concept (lemma), like when comparing "ran" vs "running", we use "run".
    - **Fuzzy match:** fuzzy matching is used with a threshold.  
    - **Semantic similarity:** If fuzzy matching fails, a transformer is used to calculate whether the predicted items are close to the ground truth.  
    
### Metrics
After matching, counts are computed as follows:
- True Positive (TP) = number of matched food items
- False Negative (FN) = number of GT food items not matched
- False Positive (FP) = number of predictions not matched

Then we calculate following metrics:
- **Precision**: Of the items the model identified, how many were correct?  = TP / (TP + FP)
- **Recall**: Of all the items that were actually in the image, how many did the model find?  = TP / (TP + FN)
- **F1-Score**: The mean of Precision and Recall, giving a single score for recognition accuracy. = 2 * Precision * Recall / (Precision + Recall)


### Sample Results

For the following image: 

![sample](/file/128200.jpg)

We have the following predictions from the models (the full test run results can be seen here: [evaluation_report_detailed_sample.csv](/output/evaluation_reports/evaluation_report_detailed_sample.csv)): 

|        |                                                    |                                          |                                      |                                                    |                                                    |                                                         |                                                    |                                                              |
| ------ | -------------------------------------------------- | ---------------------------------------- | ------------------------------------ | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| image  | ground_truth_items                                 | anthropic/claude-sonnet-4_detected_items | gemini-1.5-flash_detected_items      | gemini-2.0-flash_detected_items                    | gemini-2.5-flash_detected_items                    | meta-llama/llama-3.2-11b-vision-instruct_detected_items | qwen/qwen2.5-vl-72b-instruct:free_detected_items   | mistralai/mistral-small-3.2-24b-instruct:free_detected_items |
| 128200 | bagel-without-filling, philadelphia, salmon-smoked | bagel-without-filling, salmon-smoked     | bagel-without-filling, salmon-smoked | bagel-without-filling, cream-cheese, salmon-smoked | bagel-without-filling, cream-cheese, salmon-smoked | bagel, cream cheese, smoked salmon                      | bagel-without-filling, cream-cheese, salmon-smoked | bagel-without-filling, salmon, tomato-raw                    |

From these 50 test images, we have the following performance results for all models:

| Model                                         | Avg F1 (all attempts) | Avg F1 (valid responses) | Good Results (F1≥0.8) (%) | Technical Success Rate (%) | Avg Latency (ms) |
| --------------------------------------------- | --------------------- | ------------------------ | ------------------------- | -------------------------- | ---------------- |
| gemini-2.0-flash                              | 0.60                  | 0.60                     | 34                        | 100                        | 3629.46          |
| gemini-2.5-flash                              | 0.53                  | 0.53                     | 32                        | 100                        | 9742.82          |
| qwen/qwen2.5-vl-72b-instruct:free             | 0.53                  | 0.53                     | 36                        | 100                        | 11946.22         |
| gemini-1.5-flash                              | 0.44                  | 0.44                     | 32                        | 100                        | 3334.96          |
| anthropic/claude-sonnet-4                     | 0.39                  | 0.39                     | 24                        | 100                        | 8073.86          |
| meta-llama/llama-3.2-11b-vision-instruct      | 0.33                  | 0.35                     | 18                        | 94                         | 20672.53         |
| mistralai/mistral-small-3.2-24b-instruct:free | 0.17                  | 0.30                     | 8                         | 56                         | 4543.25          |


### Takeaways
1. **Using LLMs:** In the beginning, I thought it was more about prompt engineering. Although that is important, in real life, it’s more important to choose an LLM that meets the needs. The evaluation part has been the most challenging for me.  
2. **Model selection:** When testing on a smaller batch, GPT-5-mini and Gemini-2.5-pro did not outperform other models. The real choice of model always comes down to balancing needs, efficiency, and cost. 