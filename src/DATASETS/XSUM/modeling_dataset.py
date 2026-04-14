import os,sys
from typing import List, Optional, Tuple, Union
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

class XSUM_Dataset_Wrapper:
    def __init__(self,
    tokenizer,
    dataset_name:str="EdinburghNLP/xsum",
    truncation_highlights_len:Optional[int]=None,
    truncation_article_len:Optional[int]=None,
    num_proc:Optional[int]=1, 
    batch_size:Optional[int]=5,
    SEED:Optional[int]=0,
    local_path:Optional[str]=None,
    add_prefix:Optional[str]=None):

        if local_path is not None:
            dataset_path = os.path.join(local_path, dataset_name)
            self.dataset = load_from_disk(dataset_path)
        else:
            self.dataset = load_dataset(dataset_name)


        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.add_prefix = add_prefix

        if truncation_highlights_len is None:
            self.truncation_highlights_len = tokenizer.model_max_length
        else:
            self.truncation_highlights_len = truncation_highlights_len

        if truncation_article_len is None:
            self.truncation_article_len = tokenizer.model_max_length
        else:
            self.truncation_article_len = truncation_article_len
            
        self.SEED = SEED

        # Define a mapping dictionary
        column_mapping = {
            'document': 'article',
            'summary': 'highlights'}

        # Apply the rename to all splits
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].rename_columns(column_mapping)


    def __len__(self):
        return {k: len(v) for k, v in self.dataset.items()}
    
    def __getitem__(self, split: str, idx:int):
        max_len = self.__len__()[split]
        assert 0 <= idx < max_len , f"idx must be between 0 and {max_len}"
        return self.dataset[split][idx]

    
    @staticmethod
    def _tokenize_and_length(batch,tokenizer,truncation_highlights_len:int,truncation_article_len:int,add_prefix:str)->dict:
        """
        Return tokenization of article and highlights with there length
        """
        if add_prefix is not None:
            batch["article"] = [add_prefix + article for article in batch["article"]]

        source = tokenizer(batch["article"], truncation=True, max_length=truncation_article_len, add_special_tokens=True)
        target = tokenizer(batch["highlights"], truncation=True, max_length=truncation_highlights_len, add_special_tokens=True)

        return {
            'input_ids': source['input_ids'],
            'input_mask': source['attention_mask'],
            'input_len': [len(ids) for ids in source['input_ids']],
            'target_ids': target['input_ids'],
            'target_mask': target['attention_mask'],
            'target_len': [len(ids) for ids in target['input_ids']],

        }
     
    def prepare_dataset(self)->None:
        self.dataset = self.dataset.map(lambda batch: self._tokenize_and_length(batch,
                                        self.tokenizer,
                                        self.truncation_highlights_len,
                                        self.truncation_article_len,
                                        self.add_prefix), 
                                        num_proc=self.num_proc,
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
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)

    if 'WORK' in os.environ:
        assert os.environ['WORK'] == "/lustre/fswork/projects/rech/fte/usz69xj"
        root_path_jeanzay = Path(os.getenv("WORK"))/ "DATASETS"
        xsum_dataset = XSUM_Dataset_Wrapper(tokenizer,
                                            dataset_name="xsum_dataset",
                                            truncation_highlights_len=900,
                                            truncation_article_len=1024, 
                                            batch_size=32,
                                            SEED=rd.randint(0,100),
                                            local_path=root_path_jeanzay,
                                            add_prefix="summarize: ")
        
    else:
        xsum_dataset = XSUM_Dataset_Wrapper(tokenizer) 

    # print(xsum_dataset.dataset)
    xsum_dataset.reduce_size(0.01)
    xsum_dataset.prepare_dataset()




