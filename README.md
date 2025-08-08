
Eczema AI Food Tracker: Using YOLO, Local LLM, and Gemini

## Background

Many people suffer from eczema, and while environmental and dietary factors are known to be common triggers, these triggers are highly individual. What affects one person may not affect another.

It’s helpful to build **a personal trigger database** — so that when a flare-up happens, you can trace it back to potential causes more easily.

When it comes to dietary factors, manually recording every meal is time-consuming and often inconsistent. This project simplifies this process:

You just take a photo of your food, and the system automatically detects and logs the food items. These entries become part of your personal eczema trigger history.

> Note: Currently, only the AI-based food recognition component has been implemented. Other parts are not yet included.

## Summary

The project started with a YOLO-based food detection model. While effective for basic object detection, YOLO required more setup and did not scale well for fine-grained food classification.

To improve accuracy and reduce configuration overhead, the project transitioned to using vision-capable Large Language Models (LLMs). These models offer broader generalization and require less custom training.

For those interested in the development process, the original YOLO implementation is still included in the project files.

## Dataset

This project uses the [Food Recognition 2022 dataset](https://datasetninja.com/food-recognition).

The dataset contains 43,962 images with 95,009 labeled objects across 498 food classes.

## Model Details

### YOLO Training



### Local LLM


### Gemini Integration
