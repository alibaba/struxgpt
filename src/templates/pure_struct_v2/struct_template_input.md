Your task is to analyze the given sentences to extract and represent the intrinsic semantic hierarchy systematically.

Follow this approach to ensure clarity and utility in your analysis:
1. **Comprehension**: Begin with a thorough reading to understand the overarching theme of the input sentences.
2. **Defining Scope**: Summarize the central theme to establish the scope of the semantic analysis.
3. **Aspect Breakdown**: Identify the core aspects of the discussion. For any aspect with additional layers, delineate "SubAspects" and repeat as necessary for complex structures. Each aspect or subaspect should be highly summarized and self-contained.
4. **Mapping**: Assign sentence numbers to their respective aspects or subaspects, indicating where in the text they are addressed.

Structure your analysis in a YAML format according to this template, and ensure the format is clean, well-organized, and devoid of extraneous commentary:
```yaml
Scope: <central theme summary>
Aspects: 
  - AspectName: <main aspect>
    SentenceRange: 
      start: <start sentence number>
      end: <end sentence number>
    SubAspects: 
      - AspectName: <subaspect>
        SentenceRange:
          start: <start sentence number>
          end: <end sentence number>
        # Recursively repeat "SubAspects" structure as needed
      # Adjust "SubAspect" entries as needed
  # Adjust "Aspect" entries as needed
```

Now, analyze the provided sentences with the structured analytical process, and output your analysis in the structured YAML format.
NOTE: each aspect or subaspect should be highly summarized and self-conatined, which covers at least two sentences, except for introduction or conclusion aspects.

### Input Sentences:
```
{sentences}
```

### Output Analysis:
[Your YAML-formatted analysis goes here]