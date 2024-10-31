# Structure-aware Domain Knowledge Injection for Large Language Models

This project focuses on efficient and effective (domain) knowledge injection. Our method is evaluated by two settings:
1. free-form closed-book question answering on LongBench, which requires factual knowledge to generate answers without context references;
2. multi-choice question answering on MMedBench, which involves practical diagnoses using medicine knowledge across 6 languages.

Below is corresponding scripts to experiment on those benchmarks. Both follow the Continual Pre-Training (CPT) and Supervised Fine-Tuning (SFT) paradigm for knowledge injection and instruction alignment respectively.

Click a link below to see the detailed instructions to reproduce the results.

* [Free-Form Evaluation on LongBench](examples/StructTuning/LongBench/README.md)
* [Multi-Choice Evaluation on MMedBench](examples/StructTuning/MMedBench/README.md)
