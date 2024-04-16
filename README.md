# Using Adaptive Empathetic Responses for Teaching English
Code for our BEA 2024 Submission: Using Adaptive Empathetic Responses for Teaching English.

![System Structure](imgs/figure_1.png)

## Repository Structure
The repository is organized as follows:
- `api_server/`: The code used to run the API server for the bot on the EduBot platform. It is modularized such that you would be able to swap out different components of our pipeline. 
  - Our code for identifying specific type of grammar error given the correction (e.g. narrowing it down to a word order error) is not publicly available, so we have chosen to replace it with direct rephrasing (i.e. directly repeating the corrected user's utterance based on our grammar correction model output). You can replace it with other approaches such as a compounded call to LLMs.
  - We only provide the API server and don't provide the frontend architecture. 
- `audio_emotion_data/`: This is the audio clips we have manually labeled as `Neutral`, `Negative`, `Pauses`, or `Unusable` as specified in the paper.
  - We will be releasing audio clips with verified ASR transcripts around August 2024 after removing all identifiable information.
- `dspy_generations/`: Data we used and the corresponding generations for different conditions for our adaptive empathetic feedback module using DSPy. 

## Training Grammar Models

## Testing Modules
For each component of the pipeline, you would be able to run the files directly to test the functionalities.
