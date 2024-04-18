import glob
import os.path

import flask
import asyncio

import librosa
from flask import Flask, render_template, request, session
from flask_session import Session
from flask_cors import CORS
import dspy
class StudentFeedback(dspy.Signature):
    """A student is learning English. You are assessing a spoken utterance. In at most two sentences, summarize (1) their specific strengths in English skills and (2) things they can work on to improve. Address the student in the second person. Include specific examples that the student can learn from. Be colloquial, as if in spoken conversation."""

    convo = dspy.InputField()
    output = dspy.OutputField(desc="Treat this as a spoken conversation, so be succinct, colloquial, and empathetic.")

class OfferFeedback(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_feedback = dspy.ChainOfThought(StudentFeedback)

    def forward(self, convo):
        answer = self.generate_feedback(convo=convo)
        return answer
from empathy_generation import OfferFeedback, StudentFeedback, call_empathy_gen
from ehcalabres_wav2vec_zeroshot import call_frustration
import logging
import argparse

import sys
import random
import requests
from query_response import classify_query, respond_to_user


def send_for_response(text, history):
    raise NotImplementedError

def provide_grammar_correction(text):
    raise NotImplementedError



parser = argparse.ArgumentParser(description="Simple API for chat bot")
parser.add_argument('--serving_hostname', default="0.0.0.0", help="API web server hostname.")
parser.add_argument('--serving_port', type=int, default=8080, help="API web server port.")

args = parser.parse_args()

serving_hostname = args.serving_hostname
serving_port = args.serving_port


# Create the Flask app instance
app = Flask(__name__)

LOGGER = logging.getLogger('gunicorn.error')

SECRET_KEY = 'YOURKEY'
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)

Session(app)
CORS(app)
blueprint = flask.Blueprint('parlai_api', __name__, template_folder='templates')

import json
ERROR_REPHRASES = json.load(open("utterances/error_rephrase.json"))["rephrasers"]


FRUST_THRESHOLD = 0.5

empathy_response_storage = {}
grammar_feedback_storage = {}
feedback_buffer = {}


# Define a route for the root URL
@blueprint.route('/api/v1/call', methods=["POST"])
def call_empathy_responses():
    data = request.get_json()

    text, history, audio_url, uid = data.get('user_text', None), data.get('updated_hist', []), data.get('audio_url', None), data.get("uid", "")
    text = text + "\n\n"

    # If we have exceeded 10 turns, we say the conversation is now over
    # Note that the current version does not include the feedback turns or user inquiries after the feedback
    if len(history) >= 20:
        ep_done = True
    else:
        ep_done = False

    if uid in feedback_buffer and feedback_buffer[uid]:
        if classify_query(text) and len(history) > 3:
            query_resp = respond_to_user(history[-2], history[-1], text)
            return {
                "response": query_resp,
                "updated_hist": history,
                "episode_done": ep_done
            }
        texts = feedback_buffer[uid].split(" | ")

        if "thank" in text.lower():
            prefix = random.choice(["Of course!", "No problem at all.", "Yeah, no problem!", "No problem!"]) + " " + random.choice(["Back to the conversation.", "Back to our convo.", "Let's go back to chatting.", "Now we circle back."])
        else:
            prefix = random.choice(
                ["Sounds great.", "Alright, let's continue our conversation.", "Great, let's get back to it!",
                 "Okay let's go back to our conversation.", "Now back to our conversation.", "Okay!",
                 "Lets' go back to our chat.", "Let's keep chatting."])

        text, vicuna = texts[0], texts[1]
        feedback_buffer.update({uid: False})

        return {
            "response": prefix + " " + vicuna,
            "updated_hist": history + [text, vicuna],
            "episode_done": ep_done
        }


    response_vicuna = send_for_response(text, history)

    if audio_url == "":
        audio_url = None
    frust, _ = call_frustration(audio_url)
    print(frust, ">>> FRUSTRATION LEVEL")

    if uid not in empathy_response_storage:
        empathy_response_storage.update({uid: -1})
    else:
        empathy_response_storage[uid] = empathy_response_storage[uid] - 1
    if uid not in grammar_feedback_storage:
        grammar_feedback_storage.update({uid: -1})
    else:
        grammar_feedback_storage[uid] = grammar_feedback_storage[uid] - 1

    if frust < FRUST_THRESHOLD or empathy_response_storage[uid] > 0:
        if grammar_feedback_storage[uid] > 0:
            grammar_correct = ""
        else:
            grammar_correct = provide_grammar_correction(text)
            grammar_feedback_storage.update({uid: 2})
        empathetic_response = ""
    else:
        # Only provide grammar correctness feedback if there is no need for empathetic feedback
        grammar_correct = ""
        empathetic_response = call_empathy_gen(history)
        empathy_response_storage.update({uid: 4})

    concat_resp_string = None
    if len(grammar_correct) or len(empathetic_response):
        feedback_buffer.update({uid: text + " | " + response_vicuna["response"]})
        concat_resp_string = grammar_correct + "  " + empathetic_response + "  " + random.choice(["How does that sound?", "Does that sound alright to you?", "", "Does that sound good?"])
        concat_resp_string = concat_resp_string.strip(" ").replace("    ", "  ")
    else:
        feedback_buffer.update({uid: False})

    # concat_resp_string = grammar_correct + "  " + empathetic_response + "  " + response_vicuna["response"]
    # concat_resp_string = concat_resp_string.strip(" ").replace("    ", "  ")

    if concat_resp_string:
        return {
            "response": concat_resp_string,
            "updated_hist": history,
            "episode_done": ep_done
        }
    else:
        return {
            "response": response_vicuna["response"],
            "updated_hist": history + [text, response_vicuna["response"]],
            "episode_done": ep_done
        }


@blueprint.route('/health', methods=['GET'])
def get_health():
    return "OK"


async def main():
    app.register_blueprint(blueprint)
    app.run(host=serving_hostname, port=serving_port)

main_loop = asyncio.get_event_loop()
main_loop.run_until_complete(main())