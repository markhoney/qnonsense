from os import system
from transformers import pipeline
import json, pyttsx3
engine = pyttsx3.init()
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', device=0)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=-1)
i = 1
while i < 6:
	intext = input("\n\nInput text:")
	outjson = generator((intext), do_sample=True, max_length=100)
	outtext = json.loads(json.dumps(outjson[0]))["generated_text"]
	print(outtext)
	engine.say(outtext)
	engine.runAndWait()
