from happytransformer import GENSettings, GENTrainArgs, HappyGeneration

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M") # 125M, 1.3B, 2.7B

# args = GENTrainArgs(learning_rate =1e-5, num_train_epochs = 1)
# happy_gen.train("posts.json", args=args)

args = GENSettings(no_repeat_ngram_size=2, do_sample=True, early_stopping=False, top_k=50, temperature=0.7)
result = happy_gen.generate_text(">>13682389\nTHE WORLD OF DEMOCRACY AND SOCIALISM.\nTRUST THE PEOPLE.\nUNITE THE WORLD.", args=args)

# print(result)
print(result.text)
