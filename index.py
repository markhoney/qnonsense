import os
# import torch
# torch.cuda.empty_cache()
from happytransformer import GENSettings, GENTrainArgs, HappyGeneration

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M") # 125M, 1.3B, 2.7B
happy_gen.gpu_support = "cpu"

preprocessed = "preprocessed-data.json"

if os.path.exists(preprocessed):
	args = GENTrainArgs(load_preprocessed_data = True, load_preprocessed_data_path = "data/preprocessed-data.json")
	happy_gen.train("10-0.txt", args = args)
else:
	args = GENTrainArgs(num_train_epochs = 1, save_preprocessed_data = True, save_preprocessed_data_path = "data/preprocessed-data.json") # learning_rate = 1e-5, , batch_size = 1
	happy_gen.train("10-0.txt", args = args)

args = GENSettings(no_repeat_ngram_size = 2, do_sample = True, early_stopping = False, top_k = 50, temperature = 0.7)
result = happy_gen.generate_text(args = args)

# print(result)
print(result.text)
