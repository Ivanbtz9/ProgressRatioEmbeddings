import os
from typing import List, Optional, Tuple, Union
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

class SQUAD_Dataset_Wrapper:
    def __init__(self,
    tokenizer,
    dataset_name:str="rajpurkar/squad",
    truncation_answer:Optional[int]=None,
    truncation_content:Optional[int]=None,
    num_proc:Optional[int]=1,
    load_from_cache_file:Optional[bool]=True, 
    batch_size:Optional[int]=20,
    SEED:Optional[int]=0,
    local_path:Optional[str]=None,
    ):

        if local_path is not None:
            dataset_path = os.path.join(local_path, dataset_name)
            self.dataset = load_dataset(dataset_path)
        else:
            self.dataset = load_dataset(dataset_name)

        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.load_from_cache_file = load_from_cache_file

        if truncation_answer is None:
            self.truncation_answer = tokenizer.model_max_length
        else:
            self.truncation_answer = truncation_answer

        if truncation_content is None:
            self.truncation_content = tokenizer.model_max_length
        else:
            self.truncation_content = truncation_content

            
        self.SEED = SEED

    def __len__(self):
        return {k: len(v) for k, v in self.dataset.items()}
    
    def __getitem__(self, split: str, idx:int):
        max_len = self.__len__()[split]
        assert 0 <= idx < max_len , f"idx must be between 0 and {max_len}"
        return self.dataset[split][idx]

    
    @staticmethod
    def _tokenize_and_length(batch,tokenizer,truncation_answer:int,truncation_content:int)->dict:
        """
        Return tokenization of question + content and answer with there length
        """
 
        # question_content = ["Question: "+ q + tokenizer.sep_token + "Content: "+ c for q, c in zip(batch["question"], batch["context"])]
        # answer = [a["text"][0] for a in batch["answers"]]

        answer_content = ["Answers: "+ a["text"][0]  + tokenizer.sep_token  + "Content: "+ c for a, c in zip(batch["answers"], batch["context"])]
        question = [q for q in batch["question"]]


        source = tokenizer(answer_content, truncation=True, max_length=truncation_content, add_special_tokens=True)
        target = tokenizer(question, truncation=True, max_length=truncation_answer, add_special_tokens=True)

        return {
            'answer_content':answer_content,
            'input_ids': source['input_ids'],
            'input_mask': source['attention_mask'],
            'input_len': [len(ids) for ids in source['input_ids']],
            'target':question,
            'target_ids': target['input_ids'],
            'target_mask': target['attention_mask'],
            'target_len': [len(ids) for ids in target['input_ids']],

        }
     
    def prepare_dataset(self)->None:
        self.dataset = self.dataset.map(lambda batch: self._tokenize_and_length(batch,
                                        self.tokenizer,
                                        self.truncation_answer,
                                        self.truncation_content,
                                        ), 
                                        num_proc=self.num_proc,
                                        load_from_cache_file=self.load_from_cache_file,
                                        batched=True, 
                                        batch_size= self.batch_size)
        
    def reduce_size(self,percentage:float):

        assert 0 < percentage <= 1 , "percentage must be in (0,1]"
        
        for name in self.dataset: 
            size = int(len(self.dataset[name]) * percentage)
            self.dataset[name] = self.dataset[name].shuffle(seed=self.SEED).select(range(size))
    
    def drop_keys(self,list_key_to_drop:list):
        self.dataset = DatasetDict({key: ds for key, ds in self.dataset.items() if key not in list_key_to_drop })


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from pathlib import Path
    import random as rd
    
    _CHECKPOINT_FOR_DOC = "facebook/bart-large"

    if 'DSDIR' in os.environ:
        assert os.environ['DSDIR'] == "/lustre/fsmisc/dataset"
        root_path_jeanzay = os.environ['DSDIR'] + '/HuggingFace_Models/'
        tokenizer = AutoTokenizer.from_pretrained(root_path_jeanzay + '/' + _CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)
        # Load CNN/DailyMail dataset
        local_path = os.environ['DSDIR'] + '/HuggingFace'
        quad_dataset = SQUAD_Dataset_Wrapper(tokenizer,
                                          dataset_name="squad", 
                                          truncation_answer=900,
                                          truncation_content=1024, 
                                          batch_size=32,
                                          SEED=rd.randint(0,100),
                                          local_path=local_path)   
    else:
        tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)
        squad_dataset = SQUAD_Dataset_Wrapper(tokenizer,
                                          dataset_name="squad", 
                                          truncation_answer=1000,
                                          truncation_content=1024, 
                                          batch_size=32,
                                          SEED=rd.randint(0,100),
                                          )
    # LOAD model weigths

    print(tokenizer.sep_token )

    squad_dataset.reduce_size(0.001)
    squad_dataset.prepare_dataset()

    print(squad_dataset.dataset)

    print(squad_dataset.dataset["train"][2])