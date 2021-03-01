import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)


# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])

masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-cased')

with torch.no_grad():
    print(tokens_tensor)
    print(segments_tensors)
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'Jim'
print(predicted_token)


# question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
# question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# # The format is paragraph first and then question
# text_1 = "Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet."
# text_2 = "Who is current CEO ?"
# indexed_tokens1 = question_answering_tokenizer.encode(text_1, add_special_tokens=True)
# indexed_tokens2 = question_answering_tokenizer.encode(text_2, add_special_tokens=True)

# indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# segments_ids = [0]*len(indexed_tokens1)
# segments_ids.extend([1]*(len(indexed_tokens2)-1))

# segments_tensors = torch.tensor([segments_ids])
# tokens_tensor = torch.tensor([indexed_tokens])

# # Predict the start and end positions logits
# with torch.no_grad():
#     out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# # get the highest prediction
# answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
# # assert answer == "puppeteer"
# print("answer", answer)

# # Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
# start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
# multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)

# a= 1