"""
Based on the user's following utterance: " I would like to talk a movie that I have watched several days ago which is called Pegasus the second and is directed by a Chinese director Hang Han. The main character of the movie was a very famous super racing driver and in his last driving race competition he after crossed the finish line he had a very serious car accident and one of the important mechanical parts of his car has been lost which make it impossible to confirm whether his result was legal. So therefore his results was cancelled and he also was seriously injured in that car accident and could not drive a race car anymore. So he had to make a living as a driving school teacher until one day the owner of a car factory found him and wanted to sponsor him to form a team to play in a competition again. The owner of the car factory wants to promote his car factory and the main character wants to trade again for his own dream so they had a deal and collaboration and after all the result proved himself to the world again.", answer the user's following query: "Am I making grammar mistakes?" Answer in a spoken utterance. Provide specific feedback, but be succinct.

"""
import openai
import numpy as np


def classify_query(user_utt):
    if "?" in user_utt:
        if np.any([x in user_utt for x in ["grammar", "grammatical", "vocab", "English", "mistake", "example", "sentence"]]):
            return True
    return False


def respond_to_user(user_utt, prev_bot_resp, user_query):
    convo = f"User: {user_utt}\nAssistant: {prev_bot_resp}\n"
    client = openai.OpenAI(api_key="OPENAI_KEY")
    prompt = f"""Based on the following conversation history:\n\n{convo}, answer the user's following query: \"{user_query}\" Answer in a spoken utterance. Provide specific feedback, but be succinct."""
    msgs = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=msgs
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(respond_to_user(" Actually, yes, there's one movie called Only the River Flows that I have watched last year. This movie was based on a book written by Yu Hua. This book told a story that one of a murder has happened in a small town and one of the police officers needs to find out who is the murderer. We only know one clue is that it may be did by a woman with long black hair. That is the only clue. So the police officer had several... How to say 嫌疑人", "May I suggest the rephrase River Flows that I watched last year. . You seem to have included an unneeded word related to verb tense in this sentence. In this context, you should drop the \"have\" before \"watched\".  Does that sound good?", " Can you please make an example, please?"))