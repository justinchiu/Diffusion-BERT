import datasets
import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence


class DiffusionLoader:
    def __init__(self, tokenizer, max_length=128, wrap_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # set truncate = False for wrapping
        self.wrap_text = wrap_text

    def _load(self, task_name, split):
        dataset = datasets.load_dataset(task_name, split=split)
        #print(f'Example in {split} set:')
        #print(dataset[0])
        dataset = dataset.map(
            partial(
                self.convert_to_features,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                wrap_text = self.wrap_text,
            ),
            batched=True,
            #remove_columns='text',
            remove_columns=dataset.column_names,
        )
        return dataset

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer, max_length, wrap_text):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['text'],
            max_length=max_length if not wrap_text else None,
            truncation=not wrap_text,
            add_special_tokens=False,
        )
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }
        if wrap_text:
            input_ids = []
            attention_mask = []
            block_size = max_length
            for tokens, mask in zip(
                input_encodings["input_ids"],
                input_encodings["attention_mask"],
            ):
                total_length = len(tokens)
                #total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                input_ids += [
                    tokens[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                attention_mask += [
                    mask[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            encodings = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        return encodings

class ConditionalLoader:
    def __init__(self, tokenizer, return_source_length=False):
        self.tokenizer = tokenizer
        self.return_source_length = return_source_length
        self.data_dir = './conditional_data'

    @staticmethod
    def _convert_to_features_original(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=False)
        return {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

    def load_original(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        dataset = dataset.map(partial(self._convert_to_features_original, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def _load(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        if self.return_source_length:
            dataset = dataset.map(partial(self.add_original_src_length, tokenizer=self.tokenizer))
        dataset = dataset.map(self.add_prompt)
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def add_original_src_length(self, example, tokenizer):
        return {
            'original_src_length': len(tokenizer.encode(example['src'], max_length=128, truncation=True, add_special_tokens=False))
        }

    def my_load(self, splits):
        return [self._load(name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=False)
        encodings = {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

        return encodings

    @staticmethod
    def collate_fn(batch_input, tokenizer):
        input_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source'] + d['target'] + [tokenizer.sep_token_id]
        ) for d in batch_input], batch_first=True)

        attention_mask = torch.ones_like(input_ids)

        target_mask = torch.stack([torch.cat([
            torch.zeros(len(d['source']) + 1), torch.ones(input_ids.size(1) - len(d['source']) - 1)
        ]) for d in batch_input])

        assert input_ids.size() == attention_mask.size() == target_mask.size()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

class QQPLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QQPLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'qqp'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" is equal to "'
        example['trg'] = example['trg']
        return example



class QTLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QTLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'Q-T'

    @staticmethod
    def add_prompt(example):
        example['src'] = ' Answer: ' + example['src'] + ' Question: '
        example['trg'] = example['trg']
        return example


class WikiLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(WikiLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'wiki_alignment'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" can be summarized as: '
        example['trg'] = example['trg']
        return example

class CCLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(CCLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'CC'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src'] + ' - '
        example['trg'] = example['trg']
        return example


class DiffusionLoaderWithElectra(DiffusionLoader):
    def __init__(self, model_tokenizer, electra_tokenizer, electra_model):
        super().__init__(model_tokenizer)
        self.electra_tokenizer = electra_tokenizer
        self.electra_model = electra_model

    def _load(self, task_name, split):
        dataset = datasets.load_dataset(f'./dataloaders/{task_name}.py', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.new_convert_to_features, model_tokenizer=self.tokenizer, electra_tokenizer=self.electra_tokenizer, electra_model=self.electra_model), batched=True, remove_columns='text')
        return dataset

    @staticmethod
    def new_convert_to_features(example_batch, model_tokenizer, electra_tokenizer, electra_model):
        input_encodings = model_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, add_special_tokens=False)
        electra_encodings = electra_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, padding=True, return_tensors='pt', add_special_tokens=False)
        for k in electra_encodings.keys():
            electra_encodings[k] = electra_encodings[k].cuda()
        position = electra_encodings['attention_mask'].count_nonzero(1)
        with torch.no_grad():
            logits = electra_model(**electra_encodings)


        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'electra_logits': [logits[i][:position[i]] for i in range(position.size(0))]
        }

        return encodings


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader = DiffusionLoader(tokenizer, max_length=2048, wrap_text = True)
    data = loader.my_load("pg19", ["validation"])
    import pdb; pdb.set_trace()
