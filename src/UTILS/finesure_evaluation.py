'''
This is for providing fundamental functions for FineSurE.
'''
import ast
import torch
import sys 

ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']


def get_response(messages_batch, tokenizer, model, MAX_CONTEXT_LEN=4096, temperature=0.0, max_new_tokens=5000,rank=0):
    ''' 
    Processes messages one by one to avoid padding issues.
    '''
    
    # Manually toggle backends if context manager isn't available
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    text_responses = []
    
    # Process each conversation in the "batch" individually
    for i, messages in enumerate(messages_batch):
        print(f"Rank {rank} - Generating for sample {i+1}/{len(messages_batch)}")
        
        # 1. Apply template without padding (not needed for single sequence)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,           
            max_length=MAX_CONTEXT_LEN, 
        ).to(model.device)

        # 2. Generation
        do_sample = temperature > 0
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None, 
                top_k=50 if do_sample else None,
                pad_token_id=tokenizer.eos_token_id 
            )
        
        # 3. Slice and Decode
        # prompt_length is the length of input_ids for THIS specific sequence
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0, prompt_length:] 
        
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        text_responses.append(response)

    return text_responses


'''
Function for factuality checking 
'''

def get_fact_checking_prompt(input, sentences):
    
    ''' A function to define the input prompt
    Args:
        input: input document (reference article)
        sentences: list of summary sentences
    Return: 
        prompt: the final input prompt
    '''

    num_sentences = str(len(sentences))
    sentences = '\n'.join(sentences)

    prompt = \
"""
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}, {"sentence": "third sentence", "reason": "your reason", "category": "entity error"},]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input, num_sentences, sentences)

    return prompt


'''
Parsing the llm content
'''

def parsing_llm_fact_checking_output(output):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return: 
        pred_labels: the binary label for each sentence (0: no factuality error, 1: factuality error)
        pred_types: the error type of each sentence 
    '''

    try:
        start_idx = output.find('[')

        if start_idx != -1:
            end_idx = output.find(']')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            for out in output:
                category = out["category"]
                category = category.replace('\n', '').replace('[', '').replace(']', '')
                if category.lower() == "no error":
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
                pred_types.append(category)
            return pred_labels, pred_types
        
        else:
            start_idx = output.find('{')
            end_idx = output.find('}')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            category = output["category"]
            category = category.replace('\n', '').replace('[', '').replace(']', '')
            if category.lower() == "no error":
                pred_labels.append(0)
            else:
                pred_labels.append(1)
            pred_types.append(category)
            return pred_labels, pred_types
        
    except Exception as e:
        
        try:
            subseqs = output.split("category")

            def error_detection(subseq):
                detected = False
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        detected = True
                        detected_type = error_type
                if detected:
                    return 1, error_type
                else:
                    return 0, "no error"
                
            pred_labels, pred_types = [], []
            for subseq in subseqs:
                error_label, error_type = error_detection(subseq)
                pred_labels.append(error_label)
                pred_types.append(error_type)
        
            return pred_labels, pred_types
        
        except Exception as e:
            print('parsing error:', e)
            return [], []


'''
 Score funtions
'''
def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    faithfulness = 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)  
    return faithfulness
