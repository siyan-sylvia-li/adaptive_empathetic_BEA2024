# Using DSPy to Generate Adaptive Empathetic Feedback

### Files in this directory

* `adaptive_empathetic_dspy.ipynb`: This Jupyter notebook details how we optimized the framework to create adaptive empathetic feedback.
  * You can upload this notebook, along with `tvt_final.json`, to Google Colab and run it there. You may need to restart the session to make sure the DSPy import runs smoothly.
* `dspy_generations_data.json`: This contains the data we used for the user study that is discussed in the paper. 
  * "convo" refers to the three utterances fed into the adaptive empathetic feedback pipeline.
  * "zeroshot" is the output from the zeroshot stage (initial prompts).
  * "optimized" is the output from using the prompt optimized with DSPy using BayesianSignatureOptimizer.
  * "rewrite" is the result from "rewriting" the "optimized" output to shorten and colloquialize the generation.
* `emp_bot.json`: The optimized DSPy prompt that can be reloaded. Please refer to the next section for how to use it for your own application.
* `tvt_final.json`: The data we use to optimize our prompts. The version here is an anonymized version that excludes any mention of names, therefore the numbers in the paper may not be fully reproducible. There should still be a significant improvement in GPT-4 scores after optimization.

### Reloading the Optimized Prompt
After copying over the definitions for `OfferFeedback` and `StudentFeedback` from `api_server/empathy_generation.py`, you can reload the prompts using the following code. 
```python
import dspy
turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=1000)
dspy.configure(lm=turbo)
reload_chain = OfferFeedback()
reload_chain.load("emp_bot.json")
```

To call the reloaded prompts, you would first need three consecutive utterances, and then format them into the following format:

```
- utterance 1
- utterance 2
- utterance 3
```

Upon formatting the conversation, you can proceed to use the prompt like this.

```python
conv = "<formatted last three utterances>"
generation = reload_chain.forward(conv).output
```

You can refer to `api_server/empathy_generation.py` for the rewrite stage.

