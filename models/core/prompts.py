FOOD_DETECTION_PROMPT = """
You are an expert food identification AI. Analyze the provided image and identify ALL specific food items using the exact subcategory names from the comprehensive list below. Your entire output must be the raw JSON.

## CRITICAL RULES:
1. **Analyze the image thoroughly** and identify ALL food items you can see
2. Your response MUST be a valid JSON object with NO extra text or markdown
3. **Use EXACT subcategory names** from the list below - be as specific as possible
4. **If you see multiple items**, list them all
5. **Look for both prepared dishes AND individual ingredients**
6. **Prioritize specificity** - choose the most specific subcategory that matches what you see
7. **Be conservative but thorough** - only include items you can clearly identify

## COMPLETE SUBCATEGORY LIST (organized by category):
{subcategory_text}

## JSON RESPONSE FORMAT:
{{
    "detected_foods": [
        {{
            "subcategory": "bread-sourdough",
            "confidence": "high",
            "reasoning": "Clear sourdough bread with characteristic thick crust and open crumb structure"
        }},
        {{
            "subcategory": "cheddar",
            "confidence": "medium", 
            "reasoning": "Yellow cheese slice, appears to be cheddar based on color and texture"
        }}
    ],
    "overall_confidence": "high",
    "description": "Sandwich with sourdough bread and cheddar cheese clearly visible"
}}

## VISUAL ANALYSIS TIPS:
- **Texture**: smooth, rough, crispy, soft, creamy, grainy, fibrous
- **Color**: exact shades, natural vs processed coloring
- **Shape**: specific forms, cuts, natural vs manufactured
- **Size**: grain size, piece size, relative proportions  
- **Preparation**: raw, cooked, grilled, fried, steamed
- **Context**: how is it presented, mixed with other ingredients?

## COMMON IDENTIFIERS:
- **Bread types**: Look for crust thickness, color, texture, seeds
- **Rice varieties**: Grain length, color (white/brown), texture
- **Cheese types**: Color, texture, holes, aging marks
- **Vegetables**: Fresh vs cooked, leaf shape, color intensity
- **Meat types**: Color, texture, cut, preparation method

## TASK:
Analyze the provided image and return ONLY the raw JSON object detailing your findings.
"""


FOOD_POPULATE_PROMPT = """
You are an expert food ontologist creating ground truth data for image recognition systems. For each food item provided, perform these two tasks:

1. **Classify**: Choose the most appropriate parent category from this list:
{parent_categories}

2. **Generate Similar Items**: Create terms that could be confused with or matched to this food item in image recognition, prioritized as follows:

   **Priority 1 - Visual Similarity (Most Important)**:
   - Items with similar shape, size, color, or texture in photographs
   - Foods that look alike when prepared the same way
   - Common visual misidentifications (e.g., "lime" → "lemon", "zucchini" → "cucumber")
   
   **Priority 2 - Linguistic Variants**:
   - Synonyms & regional names (e.g., "eggplant" → "aubergine")
   - Common alternative names (e.g., "scallion" → "green onion", "spring onion")
   
   **Priority 3 - Preparation Variants**:
   - Same food in different preparation states (e.g., "carrot" → "raw carrot", "diced carrot")
   - Common cooking variations if visually distinguishable
   
   **Include sparingly - Context-Specific**:
   - Generalizations only if they aid image classification (e.g., "cheddar" → "cheese")
   - Common misspellings only if frequently encountered in image datasets

**Rules for Similar Items:**
- All items must be lowercase
- Do not include the original food item name
- Maximum 5 similar items per food (prioritize visual similarity)
- Use empty list [] if no suitable items found
- Ask: "Could a person or AI confuse these visually in a photo?"
- Avoid items that are functionally similar but visually distinct

**Food Items to Process:**
{food_items}

**Required JSON Response Format:**
{{
  "item-name-1": {{
    "parent_category": "Category Name",
    "similar_items": ["visual_match1", "visual_match2", "synonym1", "preparation_variant1"]
  }},
  "item-name-2": {{
    "parent_category": "Category Name", 
    "similar_items": ["visual_match3", "visual_match4"]
  }}
}}

Focus on creating reliable ground truth for image recognition - prioritize what would actually be confused in photographs over abstract food relationships.
"""

FOOD_SYNTHESIZE_PROMPT = """
You are an expert food taxonomist creating the definitive classification for "{item_name}" by synthesizing two expert opinions for food image recognition ground truth.

## EXPERT OPINIONS:

**EXPERT 1 (BASE):**
- Category: {base_category}
- Similar Items: {base_similar}

**EXPERT 2 (CHALLENGER):**
- Category: {challenger_category}
- Similar Items: {challenger_similar}

## SYNTHESIS METHODOLOGY:

**Category Decision Framework:**
1. If categories are identical → use that category
2. If categories differ:
   - Choose the more specific category when both are valid
   - For preparation conflicts: prioritize the preparation state that's more visually obvious
   - For hierarchical conflicts: choose based on primary visual characteristics
   - When uncertain: default to the category that better serves image recognition

**Similar Items Synthesis:**
1. **Merge and deduplicate** all similar items from both experts
2. **Score each item** by these criteria:
   - Visual similarity in photos (highest weight)
   - Linguistic correctness and common usage
   - Relevance for image recognition confusion
3. **Select top 6 items** based on scores
4. **Remove**: duplicates, self-references, non-English terms, brand names

**Quality Validation:**
- Final similar_items must be lowercase English terms
- Each item should answer: "Could this be confused with {item_name} in a photo?"
- Items should help improve image recognition accuracy

## DECISION LOGIC:
- **Agreement**: When experts agree, validate their choice
- **Disagreement**: Synthesize based on which opinion better serves image recognition
- **Quality over quantity**: Better to have 3 perfect items than 6 mediocre ones

## VALID CATEGORIES:
{parent_categories_str}

## OUTPUT REQUIREMENT:
Provide ONLY the final JSON synthesis:
{{"parent_category": "Final Category", "similar_items": ["synthesized1", "synthesized2", "synthesized3"]}}

Create the most accurate and useful classification for "{item_name}" image recognition ground truth."""


FOOD_CHALLENGER_PROMPT = """
You are a critical food classification expert tasked with reviewing and potentially challenging an existing classification for "{item_name}".

## EXISTING CLASSIFICATION TO REVIEW:
**Current Category**: {base_category}
**Current Similar Items**: {base_similar}

## YOUR TASK:
Critically evaluate the existing classification and provide an alternative if you identify issues:

**Category Challenge Criteria:**
- Is the category too broad/narrow for the item?
- Does the preparation state suggest a different category?
- Is there a more accurate category match?

**Similar Items Challenge Criteria:**
- Are the items actually visually similar in photos?
- Are important visual lookalikes missing?
- Are any items inappropriate for image recognition?
- Do items follow the visual similarity priority?

**Challenge Guidelines:**
- ONLY disagree if you have good reason
- Focus on visual similarity for image recognition context
- Consider how this item typically appears in photos

## VALID PARENT CATEGORIES:
{parent_categories_list}

## RESPONSE FORMAT:
Provide your classification (whether agreeing or challenging):
{{"parent_category": "Your Category Choice", "similar_items": ["item1", "item2", "item3", "item4", "item5"]}}

Remember: You're challenging for IMPROVEMENT, not just to be different."""

FOOD_NAME_SIMPLIFIER_PROMPT = """
You are a food naming expert. Convert the technical food item name into a natural, user-friendly display name that people would recognize.

## SIMPLIFICATION RULES:
1. **Remove technical specifications**: "without-addition-of-salt", "with-caffeine"
2. **Keep essential preparation**: "steamed", "raw", "grilled" (if it affects appearance)
3. **Use natural language**: Replace hyphens with spaces where appropriate
4. **Maintain clarity**: The result should be immediately recognizable
5. **Keep lowercase**: Consistent with your system requirements

## EXAMPLES:
- "beetroot-steamed-without-addition-of-salt" → "steamed beetroot"
- "sandwich-ham-cheese-and-butter" → "ham and cheese sandwich"  
- "coffee-with-caffeine" → "coffee"
- "tomato-raw" → "raw tomato" (keep "raw" as it's visually relevant)
- "chicken-breast-grilled" → "grilled chicken breast"

## TECHNICAL NAME: "{item_name}"

## GUIDELINES:
- Ask: "How would someone naturally describe this food?"
- Keep preparation methods that affect visual appearance
- Remove nutritional/additive specifications
- Ensure the result matches how the item appears in photos

Respond with valid JSON only:
{{"display_name": "simplified name"}}"""


FOOD_CHALLENGER_BATCH_PROMPT = """You are a critical food classification expert tasked with reviewing and potentially challenging existing classifications for multiple food items.

## YOUR TASK:
For each item below, critically evaluate the existing classification and provide an alternative ONLY if you identify issues. Your goal is IMPROVEMENT, not just being different.

## CHALLENGE CRITERIA:
**Category Challenges:**
- Is the category too broad/narrow for the item?
- Does the preparation state suggest a different category?
- Is there a more accurate category match?

**Similar Items Challenges:**
- Are the items actually visually similar in photos?
- Are important visual lookalikes missing?
- Are any items inappropriate for image recognition?
- Do items follow visual similarity priority over conceptual similarity?

## CHALLENGE GUIDELINES:
- ONLY disagree if you have good reason - agreement is perfectly valid
- Focus on visual similarity for image recognition context
- Consider how each item typically appears in photos
- Evaluate each item independently
- Prioritize visual lookalikes that would confuse image recognition

## VALID PARENT CATEGORIES:
{parent_categories_list}

## ITEMS TO REVIEW:
{items_to_review_str}

## RESPONSE FORMAT (JSON ONLY):
Your entire response must be a single JSON object with a key for each item name:
{{
  "item-name-1": {{"parent_category": "Your Category Choice", "similar_items": ["item1", "item2", "item3", "item4", "item5"]}},
  "item-name-2": {{"parent_category": "Your Category Choice", "similar_items": ["item1", "item2", "item3", "item4", "item5"]}}
}}

Remember: You're challenging for IMPROVEMENT. Agreement with existing classifications is encouraged when they are already good."""


FOOD_SYNTHESIZE_BATCH_PROMPT = """You are an expert food taxonomist creating definitive classifications by synthesizing expert opinions for food image recognition ground truth.

## SYNTHESIS METHODOLOGY:

**Category Decision Framework:**
1. If categories are identical → use that category
2. If categories differ:
   - Choose the more specific category when both are valid
   - For preparation conflicts: prioritize the preparation state that's more visually obvious
   - For hierarchical conflicts: choose based on primary visual characteristics
   - When uncertain: default to the category that better serves image recognition

**Similar Items Synthesis:**
1. **Merge and deduplicate** all similar items from both experts
2. **Score each item** by these criteria:
   - Visual similarity in photos (highest weight)
   - Linguistic correctness and common usage
   - Relevance for image recognition confusion
3. **Select top 6 items** based on scores
4. **Remove**: duplicates, self-references, non-English terms, brand names

**Quality Validation:**
- Final similar_items must be lowercase English terms
- Each item should answer: "Could this be confused with [item_name] in a photo?"
- Items should help improve image recognition accuracy

## DECISION LOGIC:
- **Agreement**: When experts agree, validate their choice
- **Disagreement**: Synthesize based on which opinion better serves image recognition
- **Quality over quantity**: Better to have 3 perfect items than 6 mediocre ones

## VALID CATEGORIES:
{parent_categories_str}

## ITEMS TO SYNTHESIZE:
{items_to_synthesize_str}

## RESPONSE FORMAT (JSON ONLY):
Your entire response must be a single JSON object with a key for each item name:
{{
  "item-name-1": {{"parent_category": "Final Category", "similar_items": ["synthesized1", "synthesized2", "synthesized3"]}},
  "item-name-2": {{"parent_category": "Final Category", "similar_items": ["synthesized1", "synthesized2", "synthesized3"]}}
}}

Create the most accurate and useful classifications for image recognition ground truth. Synthesize each item independently using the methodology above."""