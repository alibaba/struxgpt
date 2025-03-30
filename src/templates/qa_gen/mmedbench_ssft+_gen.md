You are equipped with the multilingual expertise of a medical professional and instructional designer. Your objective is to transform provided medical knowledge, structured as a mindmap, into a medical question-answer pair. This task involves logical deduction, clear reasoning, and an application of the presented medical concepts, following these directives:

Take the following steps:

1. **Interpret the Mindmap:** Review the given mindmap, focusing on crucial branches/nodes. Identify key medical concepts relevant to a clinical case scenario, such as symptoms, diagnostic findings, and physical examination highlights.

2. **Formulate a Question:** Construct a medical multiple-choice question (MCQ) with a single (or more) correct answer. Ensure the question intricately relates to the identified key concepts. Use the format provided below.
   
3. **Derive and Explain the Answer:** Clearly, logically explain the reasoning from the identified key concepts to the correct answer. Your explanation should sequentially navigate through the thought process, tying the mindmap elements to the clinical scenario and the rationale for the chosen diagnosis.

4. **Use Reference Example:** A reference question-answer (QA) example is provided. Analyze and take inspiration from its structure, clarity, and depth without directly copying its content.

5. **Structured Output:** Your explanation should be structured, concise, and easy to follow, utilizing the mindmap structure and explicitly mentioning how each element supports your conclusion.

# Input Knowledge Structure and Content

Here are relevant knowledge points regarding {field}, organized with a structurized mindmap:
```
{mindmap}
```

In the mindmap above: 
- `├` denotes branching out to a new level in the hierarchy.
- `|` marks the continuation of the current level.
- `└` indicates the last element at the current level.
- `─` connects the title of a knowledge point to its content or description.

Meanwhile, here are detailed contents regarding the specific knolwedge points:
```
{contents}
```

Your task is to design a medical question-answer pair, and logically deduce the diagnosis and explain the reasoning behind it step by step. Here is an example for reference. Do not directly copy it.

# QA Example

## Question
```
{question}
```

## Answer
```
{answer}
```

## Explanation
```
{rationale}
```

# Task Instruction

Now, design a medical question with several options for multi-choice test, and give the correct answer(s). Concurrently, give a logical and factual explanation that uses the mindmap structure and knowledge points to derive from the question to the answer. The explanation should include:
- Opening Statement: Briefly summarize the patient's clinical presentation and the task at hand.
- Integrating Mindmap Elements: Clearly state, "Based on the mindmap section on `[specific branch/node]` concerning `[specific knowledge content]`, it is evident that `[fact or rationale]`."
- Logical Deduction: Step by step, connect the dots between the mindmap elements, the patient's presentation, and the logical reasoning that leads to the diagnosis or answer. Use a narrative that is easy to follow.
- Conclusion: Conclude with a definitive statement of the diagnosis or solution, explicitly mentioning how the mindmap elements supported this conclusion.
Note: Do NOT directly output the speficified terms like "Integrating Mindmap Elements:" or "Logical Deduction:". Just state the corresponding content.

Use the same language as the provided example (in {lang}), and adopt the given symbols to format your output:

# Response

## Question
```
[The clinical scenario question goes here. Use {lang}.]
```

## Options
```
A. [Option A goes here. Use {lang}.]
B. [Option B goes here. Use {lang}.]
C. [Option C goes here. Use {lang}.]
D. [Option D goes here. Use {lang}.]
E. [Option E goes here. Use {lang}.]
[More or less options if necessary. The number of options mush align with the provided QA example.]
```

## Answer
```
[The correct option(s) index goes here, such as B for single-choice QA or A,B for multi-choice QA.]
```

## Explanation
```
[The explanation goes here. Use {lang}.]
```
