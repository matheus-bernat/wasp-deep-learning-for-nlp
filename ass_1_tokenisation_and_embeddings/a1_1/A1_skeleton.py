
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
from datasets import load_dataset


from torch.utils.data import DataLoader
import numpy as np
import sys, time, os


###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

# Task 1.2: Building the vocabulary.

def build_vocab(train_file, max_voc_size=None, pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>') -> dict[str, int]:
    """ Build a vocabulary from the given file, i.e. a mapping from token strings to integers.

        Args:
             train_file: The name of the file containing the training texts.
             max_voc_size: The maximally allowed size of the vocabulary. If None, then there is no limit.
             pad_token:  The dummy string corresponding to padding.
             unk_token:  The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:  The dummy string corresponding to the beginning of the text.
             eos_token:  The dummy string corresponding to the end the text.
        Returns:
             A dictionary mapping token strings to integers. 
             Example: {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, 'the': 4, 'a': 5, ...}
    """
    start_time = time.time()
    vocab = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
    # Read all lines into one big string.
    with open(train_file, 'r') as f:
        text = f.read()

    # Tokenize the text into a list of tokens.
    tokens = lowercase_tokenizer(text)

    # Count the frequency of each token
    counter = Counter(tokens)

    # Print some statistics about the token frequencies.
    print(f'Total number of tokens: {len(tokens)}')
    print(f'Number of unique tokens: {len(counter)}')
    print(f'Most common tokens: {counter.most_common(10)}')
    # Inspect how rare the last words in the vocab are. To judge if max_voc_size is large enough.
    if max_voc_size is not None:
        print(f'Last words in max_voc_size: {counter.most_common(max_voc_size)[-20:]}')
    # Make a plot where in the x-axis we have tokens sorted by frequency and y-axis the frequency.
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(max_voc_size), [freq for _, freq in counter.most_common(max_voc_size)])
    # plt.yscale('log')
    # plt.xlabel('Tokens sorted by frequency')
    # plt.ylabel('Frequency (log scale)')
    # plt.title('Token frequency distribution')
    # plt.show()

    # Add the most common tokens to the vocabulary, up to max_voc_size if specified.
    for token, _ in counter.most_common(max_voc_size):
        if len(vocab) == max_voc_size:
            break
        vocab[token] = len(vocab) # append to the end

    end_time = time.time()
    print(f'Vocabulary built in {end_time - start_time:.2f} seconds. Vocabulary size: {len(vocab)}.')
    
    return vocab

def get_i2t(vocab):
    # Invert the vocab dictionary to get a mapping from integers to tokens.
    i2t = {v: k for k, v in vocab.items()}
    return i2t

def get_token_id(vocab: dict[str, int], token: str, unk_token: str = '<UNK>') -> int:
    "Get the integer corresponding to the token. If the token not in vocab, return unk_token's integer."
    return vocab.get(token, vocab[unk_token])

def test_build_vocab():
    vocab = build_vocab('train.txt', max_voc_size=1000)
    i2t = get_i2t(vocab)
    assert(i2t[0] == '<PAD>')
    assert(i2t[1] == '<UNK>')
    assert(i2t[2] == '<BOS>')
    assert(i2t[3] == '<EOS>')
    # Check they don't clash with real words
    real_words = {k: v for k, v in vocab.items() if k not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']}
    assert(len(real_words) == len(vocab) - 4)
    assert(len(vocab) <= 1000)
    assert('the' in vocab)
    assert('a' in vocab)
    assert('cuboidal' not in vocab)
    assert('epiglottis' not in vocab)
    print('build_vocab test passed.')

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).

    vocab = build_vocab(train_file, max_voc_size, pad_token, unk_token, bos_token, eos_token)

    return A1Tokenizer(vocab, model_max_length, pad_token, unk_token, bos_token, eos_token)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, vocab, model_max_length=None, pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        # TODO: store all values you need in order to implement __call__ below.
        self.vocab = vocab
        self.pad_token_id = vocab[pad_token]
        self.unk_token_id = vocab[unk_token]
        self.bos_token_id = vocab[bos_token]
        self.eos_token_id = vocab[eos_token]
        self.model_max_length = model_max_length

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        # Accept either a single string or a list of strings.
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError('texts must be a string or a list of strings')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        max_length = 0
        list_of_ids = []
        for text in texts:
            # str -> list[str]
            tokens = lowercase_tokenizer(text) # "the book is on the table" -> ["the", "book", "is", ...]

            # list[str] -> list[int]
            ids = [self.bos_token_id] + \
                  [get_token_id(self.vocab, token) for token in tokens] + \
                  [self.eos_token_id]

            if len(ids) > max_length:
                max_length = len(ids)

            list_of_ids.append(ids)

        # Apply truncation and padding if specified.
        for i, ids in enumerate(list_of_ids):
            if padding and len(ids) < max_length:
                # Pad all sequences to the length of the longest sequence in the batch.
                list_of_ids[i] = ids + (max_length - len(ids)) * [self.pad_token_id]
            if truncation:
                list_of_ids[i] = list_of_ids[i][0:self.model_max_length]
            
        # Option: to be 100% HuggingFace compatible, we return an attention mask of the same shape
        # as list_of_ids. This describes which tokens are real and which are padding tokens.
        attention_mask = [[0 if id == self.pad_token_id else 1 for id in ids] for ids in list_of_ids]

        if return_tensors == 'pt':
            list_of_ids = torch.tensor(list_of_ids, dtype=torch.long)     
            attention_mask = torch.tensor(attention_mask)

        return BatchEncoding({'input_ids': list_of_ids, 'attention_mask': attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Create and save the tokenizer.
# tokenizer = build_tokenizer('train.txt', max_voc_size=10000)
# tokenizer.save(f"tokenizer.tok")

# Load the tokenizer.

def load_tokenizer():
    tokenizer = A1Tokenizer.from_file(f"tokenizer.tok")
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}.")
    return tokenizer

# Test the tokenizer.
def test_tokenizer():
    test_texts = ['This is a test.', 'Another test.']
    tokenizer = load_tokenizer()
    res = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True)
    print(res)
    # {'input_ids': tensor([[  2,  35,  14,  11, 975,   6,   3],
    #                       [  2, 155, 975,   6,   3,   0,   0]]), 
    #  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1],
    #                            [1, 1, 1, 1, 1, 0, 0]])}


###
### Part 2. Loading text files and creating batches.
###

def get_dataset(data_files={'train': 'train.txt', 'val': 'val.txt'}, use_subset=False):
    "Help function for loading dataset."
    dataset = load_dataset('text', data_files=data_files)

    print(f"Train instances: {len(dataset['train'])}")      # Train instances: 147059
    print(f"Validation instances: {len(dataset['val'])}")   # Validation instances: 17874

    # Remove empty lines
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    # Use subset for training purposes:
    from torch.utils.data import Subset
    if use_subset:
        for sec in ['train', 'val']:
            dataset[sec] = Subset(dataset[sec], range(1000))
            print(f"{sec} instances: {len(dataset[sec])} on subset") 

    
    return dataset

# TODO: optional task
# Optional task: If you want to be even more closely aligned with the HuggingFace standard API, you should also 
# 1) use tokenized texts in the Datasets instead of raw text, and 
# 2) apply a collator, such as DataCollatorForLanguageModeling.



###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=0, embedding_size=0, hidden_size=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    # Something to fix version mismatch in the transformers library between the cluster and my PC
    _tied_weights_keys = []        
    all_tied_weights_keys = {}    

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True) # batch_first: B, N, E (batch, sequence length, embedding size)
        # An output layer (aka unembedding layer) that computes the (logits of) a probability distribution over the vocaulary
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.     (B, N)
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts. (B, N)
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        loss = None

        embedded = self.embedding(input_ids) # (B, N, E)
        rnn_out, _ = self.rnn(embedded)      # (B, N, H), where H is the hidden size
        logits = self.unembedding(rnn_out)   # (B, N, V), where V is the vocabulary size
        if labels is not None:


            # Shift logits by one position to the left
            # - we don't care about the prediction for the last token, since there is no next token after it. 
            shift_logits = logits[:, :-1, :].contiguous() # contiguous makes sure the tensor is stored contiguously in memory, which is required for the loss function

            # Shift labels by one position to the right
            # - we don't care about the loss for the first token, since there is no previous token before it.
            shift_labels = labels[:, 1:].contiguous()

            # Dummy example: input = "hello friend" -> input_ids = [0, 459, 789, 1], where 0 is <BOS> and 1 is <EOS>. labels = input_ids since we want to predict the same text 
            # Then: logits = [logits(0), logits(459), logits(789), logits(1)]
            # shift_logits = [logits(0), logits(459), logits(789)] --> we don't care about the logits(1) since there is no next token after <EOS>.
            #       labels = [0, 459, 789, 1]
            # shift_labels = [459, 789, 1]   --> we don't care about the loss for the first token 0, since there is no word before <BOS>.
            # That is, we compare:
            # logits(0) -> 459 (<BOS> -> hello)
            # logits(459) -> 789 (hello -> friend)
            # logits(789) -> 1 (friend -> <EOS>)

            # Reshape logits and labels to compute the loss:
            shift_labels = shift_labels.view(-1)  # reshape the labels from (B, N) to (B*N,)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # reshape the logits from (B, N, V) to (B*N, V)
            loss = self.loss_func(shift_logits, shift_labels)

            # Example where B=2, N=3, V=4:
            # logits = [  
            #            [ [0.1, 0.2, 0.3, 0.4],   # hey -> every token in the vocab receives a probability for being the next token after "hey"
            #              [0.5, 0.6, 0.7, 0.8],   # my
            #              [0.9, 1.0, 1.1, 1.2]    # friend
            #            ],                        # batch 1,

            #            [ [0.1, 0.2, 0.3, 0.4],   # what
            #              [0.5, 0.6, 0.7, 0.8],   # is
            #              [0.9, 1.0, 1.1, 1.2]    # that
            #            ],                        # batch 2,
            #          ] # shape (batch_size, sequence_length, vocab_size)
            # After reshape:
            # logits = [ [0.1, 0.2, 0.3, 0.4],   # hey -> every token in the vocab receives a probability for being the next token after "hey"
            #            [0.5, 0.6, 0.7, 0.8],   # my
            #            [0.9, 1.0, 1.1, 1.2],   # friend
            #            [0.1, 0.2, 0.3, 0.4],   # what
            #            [0.5, 0.6, 0.7, 0.8],   # is
            #            [0.9, 1.0, 1.1, 1.2]    # that
            #          ] # shape (batch_size * sequence_length, vocab_size)

            # labels = [ 
            #            [245, 789, 1], # true labels after "hey", "my", "friend"
            #            [345, 39, 131] # true labels after "what", "is", "that"
            #          ] # shape (batch_size, sequence_length)
            # After reshape: labels = [245, 789, 1, 345, 39, 131], shape (batch_size * sequence_length,)

        # Return the original logits (without shifting or reshaping), since that's what we need for inference. We'd lose the 1-1 match between input tokens and logits
        # if we returned the shifted logits.
        return CausalLMOutput(logits=logits, loss=loss)

# Test the model on dummy data.
def test_model():
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    embedding_size = 16
    hidden_size = 32
    config = A1RNNModelConfig(vocab_size, embedding_size, hidden_size)
    model = A1RNNModel(config)

    # Create dummy input and labels.
    input_ids = torch.tensor([[2,4,5,6,3], [211,41,53,61,33]]) # B = 2, N = 5
    # input_ids = torch.tensor([[2,4,5,6,3], ]) # B = 1, N = 5

    output = model(input_ids)
    print(output.logits.shape) # should be (B*N, V) = (5, vocab_size)
    print(output)

# test_model()

###
### Part 4. Training the language model.
###


## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=True)
        
        # TODO: Your work here is to implement the training loop.

        # for each training epoch (use args.num_train_epochs here):
        for epoch in range(self.args.num_train_epochs):

            start = time.time()

            train_loss = 0
        #   for each batch B in the training set:
            for batch in train_loader:
                
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
                input_ids = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)

        #       Set the padding tokens to label -100
                labels = input_ids.input_ids.clone()
                labels[input_ids.attention_mask == 0] = -100

	    #       put input_ids and labels onto the GPU (or whatever device you use)
                input_ids = input_ids.to(device)
                labels = labels.to(device)

        #       apply the model to input_ids and labels
                output = self.model(input_ids.input_ids, labels)

        #       get the loss from the model output
                loss = output.loss

        #       BACKWARD PASS AND MODEL UPDATE:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            end = time.time()
            train_loss /= len(train_loader)
            print(f'Epoch {epoch+1}/{self.args.num_train_epochs}, Train Loss: {train_loss:.4f}, training time: {(end-start)/60:.2f} minutes.')

        #   EVALUATION:
        #   After each epoch, evaluate the model on the validation set and print the validation loss.
            val_loss = 0
            for batch in val_loader:
                input_ids = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                labels = input_ids.input_ids.clone()
                labels[input_ids.attention_mask == 0] = -100
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                output = self.model(input_ids.input_ids, labels)
                val_loss += output.loss.item()
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}/{self.args.num_train_epochs}, Validation Loss: {val_loss:.4f}, validation time: {(time.time()-end)/60:.2f} minutes.')

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    what_to_do = 'train'
    # what_to_do = 'test_network'
    if what_to_do == 'train':
        tokenizer = load_tokenizer()
        config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=16, hidden_size=32)
        model = A1RNNModel(config)

        training_args = TrainingArguments(
                optim='adamw_torch',
                use_cpu=True,
                eval_strategy='epoch',
                output_dir='trained_output',
                num_train_epochs=10,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                learning_rate=0.001)

        dataset = get_dataset(use_subset=False)

        trainer = A1Trainer(model, training_args, dataset['train'], dataset['val'], tokenizer)
        trainer.train()
    else:
        # Load model
        model = A1RNNModel.from_pretrained('trained_output')
        print('Model loaded successfully.')

        # Apply the model to the integer encoded text
        text = "She lives in San"
        tokenizer = load_tokenizer()
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        print('Input IDs:', input_ids)
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
        print('Logits shape:', output.logits.shape) # (B, N, V)

        # Use argmax to get the most likely next token for each position in the input text.
        predicted_token_ids = torch.argmax(output.logits, dim=-1) # (B, N)
        print('Predicted token IDs:', predicted_token_ids)
        # Convert the predicted token IDs back to strings using the tokenizer's vocabulary.
        i2t = get_i2t(tokenizer.vocab)
        predicted_tokens = [[i2t[token_id.item()] for token_id in batch] for batch in predicted_token_ids]
        print('Predicted tokens:', predicted_tokens)
