FOOD_DETECTION_PROMPT = """
You are an expert food identification AI. Analyze the provided image and identify ALL specific food items using the exact subcategory names from the comprehensive list below. Your entire output must be the raw JSON.

## CRITICAL RULES:
1. **Systematically scan the entire image**, from corner to corner. Identify every single food item visible, no matter how small or partially obscured. This includes main dishes, side items, ingredients, sauces, condiments, garnishes, and beverages.
2. Your response MUST be a valid JSON object with NO extra text or markdown.
3. **Use the EXACT subcategory name** from the list below for each item you identify. Do not invent new names.
4. **Composite-first approach**: If a composite dish subcategory exists in the list and is identifiable, output the composite dish. Additionally, list any ingredients that are clearly visible as distinct, separate items alongside the composite dish (not merely components integrated into the dish). If no suitable composite subcategory exists, decompose the visible dish into ingredients using exact subcategories.
5. **Prioritize specificity (with caution)**: Always choose the most specific subcategory that you can confidently identify. If you are only confident in a more general aspect (e.g., you can identify 'butter' but not 'herb-butter'), select the most general appropriate category from the list. Do not guess specifics.
6. **Be conservative but thorough**: Only include items you can reasonably identify. If uncertain about the exact type, choose the broader category but still include the item with lower confidence.
7. **Check backgrounds and edges**: Food items may be partially visible at image edges or in the background of plates/bowls.
8. **Look for layers**: In layered foods (sandwiches, lasagna, etc.), identify ingredients in each visible layer only if no composite dish category applies.

## SYSTEMATIC SCANNING PROCESS:
- **Step 1**: Identify the main/central food items
- **Step 2**: Scan all visible surfaces of plates, bowls, and containers
- **Step 3**: Check for small items: herbs, spices, seeds, nuts, berries
- **Step 4**: Look for liquids: sauces, dressings, beverages, oils
- **Step 5**: Examine any garnishes or decorative elements
- **Step 6**: Check image corners and partially visible items

## CONFIDENCE LEVELS:
- **"high"**: Item is clearly visible and easily identifiable (>90% certain)
- **"medium"**: Item is visible but some uncertainty about exact type (70-90% certain)  
- **"low"**: Item is partially obscured or difficult to distinguish but likely present (50-70% certain)

## COMPLETE SUBCATEGORY LIST (organized by category):
{subcategory_text}

## JSON RESPONSE FORMAT:
{{
    "detected_foods": [
        {{
            "subcategory": "exact-name-1-from-list",
            "confidence": "high",
            "reasoning": "brief visual justification (max ~160 chars)"
        }},
        {{
            "subcategory": "exact-name-2-from-list",
            "confidence": "medium", 
            "reasoning": "brief visual justification (max ~160 chars)"
        {{
            "subcategory": "lettuce",
            "confidence": "high",
            "reasoning": "brief visual justification (max ~160 chars)"
        }}
    ],
    "overall_confidence": "high",
    "description": "brief visual justification (max ~160 chars)"
}}

## VISUAL ANALYSIS TIPS:
- **Texture**: smooth, rough, crispy, soft, creamy, grainy, fibrous, flaky
- **Color**: exact shades, natural vs processed coloring, variations within item
- **Shape**: specific forms, cuts, natural vs manufactured, geometric patterns
- **Size**: grain size, piece size, relative proportions, thickness
- **Preparation**: raw, cooked, grilled, fried, steamed, roasted, dried
- **Context**: presentation style, mixing with other ingredients, serving method
- **Surface details**: seeds, seasonings, char marks, browning patterns

## COMMON IDENTIFIERS:
- **Bread types**: Crust thickness/color, internal texture, seeds, shape
- **Rice varieties**: Grain length, color (white/brown/wild), stickiness
- **Cheese types**: Color, texture, holes, aging marks, melting state
- **Vegetables**: Fresh vs cooked, leaf shape, color intensity, cut style
- **Meat types**: Color, texture, cut, preparation method, marbling
- **Fruits**: Ripeness, cut style, skin presence, color variations
- **Sauces**: Consistency, color, visible ingredients/particles

## TASK:
Analyze the provided image and return ONLY the raw JSON object detailing your findings. Remember to scan systematically and identify every visible food item.
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
For each item below, critically evaluate the existing classification and provide an alternative ONLY if you identify meaningful issues that would improve image recognition accuracy.

## CHALLENGE CRITERIA:
**Category Challenges:**
- Is the category too broad/narrow for the item?
- Does the preparation state suggest a different category?
- Is there a more accurate category match?

**Similar Items Challenges (Focus on Visual Recognition):**
- Are there visually similar items missing that would confuse image recognition?
- Are any current items too generic ("candy", "sweet") or irrelevant?
- Would swapping items significantly improve visual similarity?
- Are items at the right level of specificity for image recognition?
- Do items represent what people would actually see in photos?

## QUALITY STANDARDS:
- Use consistent lowercase-with-hyphens formatting: "chocolate-chip-cookie"
- Avoid overly generic terms: prefer "chocolate-candies" over "candy"
- Avoid brand names unless they represent a distinct visual category
- Focus on items that would genuinely help distinguish this food in photos
- Each similar item should answer: "Could this realistically be confused with [target] in a photo?"

## CHALLENGE GUIDELINES:
- ONLY disagree if you have substantial reasoning for improvement
- Focus on semantic/visual improvements, not just formatting consistency
- Agreement with existing classifications is encouraged when they are already good
- Consider: "Would this change actually help image recognition accuracy?"

## VALID PARENT CATEGORIES:
{parent_categories_list}

## ITEMS TO REVIEW:
{items_to_review_str}

## RESPONSE FORMAT (JSON ONLY):
{{
  "item-name-1": {{"parent_category": "Your Category Choice", "similar_items": ["item1", "item2", "item3", "item4", "item5"]}},
  "item-name-2": {{"parent_category": "Your Category Choice", "similar_items": ["item1", "item2", "item3", "item4", "item5"]}}
}}

Challenge only for MEANINGFUL improvements that enhance image recognition capability."""

FOOD_SYNTHESIZE_BATCH_PROMPT = """You are an expert food taxonomist creating definitive classifications by synthesizing expert opinions for food image recognition ground truth.

## SYNTHESIS METHODOLOGY:

**Category Decision Framework:**
1. If categories are identical → use that category
2. If categories differ:
   - Choose the more specific category when both are valid
   - For preparation conflicts: prioritize the preparation state that's more visually obvious
   - For hierarchical conflicts: choose based on primary visual characteristics
   - When uncertain: default to the category that better serves image recognition

**Similar Items Synthesis (CRITICAL FOR IMAGE RECOGNITION):**
1. **Normalize formatting**: Convert all items to consistent lowercase-with-hyphens format
2. **Merge semantically unique items**: Treat "chocolate candies" and "chocolate-candies" as the same
3. **Quality filter**: Remove overly generic terms ("candy", "sweet"), brand names (unless visually distinct), and non-descriptive items
4. **Score remaining items** by:
   - Visual similarity in photos (highest weight) - "Would this confuse image recognition?"
   - Specificity and descriptiveness for identification
   - Practical relevance for distinguishing the target food
5. **Select top 4-6 items** based on scores
6. **Final validation**: Each item should represent something that could realistically be confused with the target in a photo

**Quality Standards:**
- Prefer specific over generic: "chocolate-chip-cookies" over "cookies"
- Ensure visual relevance: items should look similar in typical food photos
- Maintain practical utility: focus on common visual confusion cases
- Use consistent formatting: lowercase-with-hyphens

## DECISION LOGIC:
- **Format-only differences**: Normalize and merge, select best version
- **Semantic disagreements**: Choose based on visual similarity and image recognition utility
- **Quality over quantity**: Better to have 4 excellent items than 6 mediocre ones
- **Visual focus**: Every item should help train better image recognition

## VALID CATEGORIES:
{parent_categories_str}

## ITEMS TO SYNTHESIZE:
{items_to_synthesize_str}

## RESPONSE FORMAT (JSON ONLY):
{{
  "item-name-1": {{"parent_category": "Final Category", "similar_items": ["specific-item1", "specific-item2", "specific-item3", "specific-item4"]}},
  "item-name-2": {{"parent_category": "Final Category", "similar_items": ["specific-item1", "specific-item2", "specific-item3", "specific-item4"]}}
}}

Create the most accurate classifications for image recognition. Prioritize visual similarity and practical utility for food identification in photos."""