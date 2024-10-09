# Text-to-Image Object Consistency Benchmark

This benchmarks aims to quantify the difference in faithfulness variability
between two object categories. By default it compares abstract vs. realistic
objects, but the principle works for other categories as well.
In other words, the benchmark checks whether the model is more consistent at
generating faithful images for one of the two categories. 

The benchmark score is the difference between the performance on the two
categories. A low absolute score is better than a high absolute score.

For more details regarding the benchmark see the [report](https://raw.githubusercontent.com/marchollinger/t2i_object_consistency_benchmark/main/paper/object_consistency.pdf).

## How to run the benchmark

The benchmark is run using `run_benchmark.py`, which takes the following arguments:

```
run_benchmark.py --test_prompts <path> --out_dir <path> --model_id <string> --score_name <string>
```

`--model_id (default="stabilityai/stable-diffusion-2")`: text-to-image model to test; a new model can be added as
long as it conforms to the interface defined in `utils.IModel`; the choices provided by default
are: "stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-1",
"stabilityai/stable-diffusion-3-medium-diffusers", "dall-e-2", "dall-e-3"; a
valid API key is required to use DALL-E

`--test_prompts (default="./abstract_vs_realistic.csv")` is a `csv` file
containing the prompts to test

`--score_name (default="align")`: similarity metric to use, can be "clip" or "align" 

`--out_dir`: directory to save results and images

After running the benchmark, the output directory will be populated with a
folder whose name is the model-id and that contains the generated images as
well as the final score and plots to visualize the result.

To produce plots to compare the results of multiple models the script
`compare_results.py` can be used to produce a box plot and a bar chart that aim
to visualize both the final benchmark score (which is the mean of all scores per
object) and the distribution of said scores.

```
compare_results.py --results_dir <path> --out_dir <path>
```

