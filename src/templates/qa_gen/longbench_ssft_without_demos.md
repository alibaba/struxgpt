You are an expert in NLP and instructional design. Your task is to create an exam question that assesses the closed-book knowledge of the mindmap’s hierarchical structure and associated passage segments. Your crafted question must elicit a response that showcases the examinee’s ability to recall and connect specific details without having access to the mindmap or passages during the exam. While designing the answer and explanation, be vigilant to explain the solution in a concise, systematic approach without inferring any creative or subjective insight or referencing instructive terms like "mindmap" or any phrases indicating direct sourcing from the study material ("in the passage").

Before constructing the question, review the following key symbols:

- `├` denotes branching out to a new level in the hierarchy.
- `|` marks the continuation of the current level.
- `└` indicates the last element at the current level.
- `─` connects the title of an element to its content or description.

For multiple passage segments, synthesize the content to pose a complex question necessitating the connection of various elements within the information provided. With a single passage segment, formulate a question that calls upon a precise detail within that particular segment.
The examination question should be answerable based on the information given in the passage and the structure of the mindmap.

Here is the mindmap structure you will use:

```
{mindmap}
```

Here are _{passage_num}_ piece(s) of segment(s) corresponding to the chosen section(s) (marked with the `**` notation) in the mindmap above:

```
{contents}
```

Use the same language as the above passages (中文 or English) to organize your response in the following format, and ensure that:

1. When there are 2 or more passage segments provided above, comprise a more sophisticated question requiring multi-hop reasoning along the mindmap structure and the provided passages. Otherwise, just derive a knowledge-intensive question targeting key detailed information.
2. The answer is direct and concise, comprising no more than 8 words.
3. Provide an explanation that logically takes one from the question to the answer, without describing how you devised the question. It should convey the chain of thought to derive the answer.

Using hashtags to separate each part:

# Question
[Formulate a SINGULAR one-hop or multi-hop question according to the given material, and do not reference instructive terms like "mindmap" or any phrases indicating direct sourcing from the study material (like "in the passage").]

# Answer
[Provide a concise answer WITHIN 8 words, which is taken directly from the provided passage and mindmap information.]

# Explanation
[A step-by-step elucidation of the one-hop or multi-hop reasoning behind deriving the answer from the question, stated objectively and avoiding any terms that indicate the behind-the-scenes creation of the material.]

Now use the target language to craft your question, answer, and explanation.