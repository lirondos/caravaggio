import torchtext
import torch
from torch import nn
from torch.distributions.uniform import Uniform
import math
from typing import Generator, Tuple
import random
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vectors

LABELS = {1: 0, 2: 1}
TRUNCATE_SEQ = 500
HIDDEN_LAYER = 400
KERNEL = 2
EPOCHS = 100
LR = 0.05
BATCH_SIZE = 50
DIMENSIONS = 300
L2 = 0.01

class DataLoader:
    def __init__(self, instances, labels) -> None:
        self.labels = labels
        self.instances = instances
        self.length = len(labels)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.instances[key], self.labels[key]
        else:
            raise TypeError('Index must be int, not {}'.format(type(key).__name__))


    def batches(
        self, batch_size: int, *, shuffle: bool
    ) -> Generator[
        Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]], None, None,
    ]:
        order = list(range(self.length))
        if shuffle:
            random.shuffle(order)

        number_of_batches = math.ceil(self.length / batch_size)
        num = 0
        while num < number_of_batches:
            my_list = [[],[]]
            for i in range(batch_size*num, batch_size*num + batch_size):
                if i< len(order):
                    my_list[0].append(self.instances[order[i]])
                    my_list[1].append(self.labels[order[i]])
            yield tuple(my_list)
            num += 1

    def pad(self, mybatch):
        max_length = 0
        instances = mybatch[0]
        labels = mybatch[1]
        for instance in instances:
            if len(instance) > max_length:
                max_length = len(instance)
        my_new_tensor_instances = list()
        my_new_tensor_labels = list()
        for i, instance in enumerate(instances):
            l = [0] * max_length
            for i, el in enumerate(instance):
                l[i] = el
            my_new_tensor_instances.append(l)
            #my_new_tensor_labels.append(labels[i].item())
        return torch.tensor(my_new_tensor_instances, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class MyEmbeddings:
    def __init__(self, embeddings, oov) -> None:
        self.oov = sorted(list(oov))
        self.itos = ["<PAD>"] + ["<UNK>"] + list(embeddings.stoi.keys()) + self.oov
        new_embeddings = torch.zeros(2 + len(embeddings.stoi) + len(oov), DIMENSIONS)
        new_embeddings[1] = Uniform(-1*math.sqrt(3/DIMENSIONS), math.sqrt(3/DIMENSIONS)).sample((1, DIMENSIONS))
        new_embeddings[2:2+len(embeddings.stoi.keys())] = embeddings.vectors
        new_embeddings[2+len(embeddings.stoi.keys()):] = Uniform(-1*math.sqrt(3/DIMENSIONS), math.sqrt(3/DIMENSIONS)).sample((len(self.oov), DIMENSIONS))
        self.embeddings = new_embeddings
        self.stoi: Dict[str, int] = {s: idx for idx, s in enumerate(self.itos)}

class DataReader:
    def __init__(self, path_to_training, original_embeddings) -> None:
        self.unique = set()
        self.oov = set()
        self.original_embeddings = original_embeddings
        with open(path_to_training, "r", encoding="utf-8") as train_file:
            for line in train_file:
                sentiment, text = line.split("\t")
                words = [word.lower() for word in text.split()[:TRUNCATE_SEQ]]
                self.unique.update(words)
                for word in words:
                    if word not in self.original_embeddings.stoi:
                        self.oov.add(word)
        print(f"Vocab: {len(self.unique)}: oov {len(self.oov)}")
        self.embeddings = MyEmbeddings(self.original_embeddings, self.oov)
        print(f"Size: {self.embeddings.embeddings.shape}")

    def read_data(self, path_to_data):
        with open(path_to_data, "r", encoding="utf-8") as train_file:
            labels = list()
            instances = list()
            for line in train_file:
                sentiment, text = line.split("\t")
                words = [word.lower() for word in text.split()[:TRUNCATE_SEQ]]
                labels.append(torch.tensor([LABELS[int(sentiment)]]))
                my_ids = [self.embeddings.stoi[word] for word in words if word in self.embeddings.stoi]
                instances.append(torch.tensor(my_ids, dtype=torch.long))
        return instances, labels

        
class LSTM(nn.Module):

    def __init__(self, embeddings) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings.embeddings)
        self.lstm = nn.LSTM(embeddings.embeddings.shape[1], HIDDEN_LAYER, batch_first = True)
        #self.conv = nn.Conv1d(embeddings.embeddings.shape[1], HIDDEN_LAYER, KERNEL, padding = KERNEL - 1)
        self.lin = nn.Linear(HIDDEN_LAYER, 2)
        self.softmax = nn.LogSoftmax(1)
        self.debug = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print("Batch shape:", batch.shape)
        embedded = self.embed(batch)
        if self.debug:
            print("Embedded shape:", embedded.shape)
        # Transpose to match conv shape
        _, (lstm_hidden, _) = self.lstm(embedded)
        lstm_squeezed = lstm_hidden.squeeze()
        if self.debug:
            print("Squeezed shape:", lstm_squeezed.shape)
        linear = self.lin(lstm_squeezed)
        output = self.softmax(linear)
        if self.debug:
            print("Output shape:", output.shape)
        return output


class CNN(nn.Module):

    def __init__(self, embeddings) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings.embeddings)
        self.conv = nn.Conv1d(embeddings.embeddings.shape[1], HIDDEN_LAYER, KERNEL, padding = KERNEL - 1)
        self.lin = nn.Linear(HIDDEN_LAYER, 2)
        self.softmax = nn.LogSoftmax(1)
        self.debug = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print("Batch shape:", batch.shape)
        embedded = self.embed(batch)
        if self.debug:
            print("Embedded shape:", embedded.shape)
        # Transpose to match conv shape
        embedd_t = embedded.transpose(1, 2)
        if self.debug:
            print("Embedded transposed shape:", embedd_t.shape)

        conved = self.conv(embedd_t)
        if self.debug:
            print("Conved shape:", conved.shape)
        # Pool over the embedding dimensions, dropping the last dimension since it's
        # now 1
        pooled = F.max_pool1d(conved, conved.shape[2])
        if self.debug:
            print("Pooled shape:", pooled.shape)
        pooled_squeezed = pooled.squeeze()
        if self.debug:
            print("Pooled squeezed shape:", pooled_squeezed.shape)
        linear = self.lin(pooled_squeezed)
        output = self.softmax(linear)
        if self.debug:
            print("Output shape:", output.shape)
        return output

if __name__ == "__main__":
    
    random.seed(0)
    path_to_training = "corpus/train.tsv"
    path_to_dev = "corpus/dev.tsv"
    
    print("Loading")
    #original_embeddings = torchtext.vocab.GloVe("6B", dim=50, cache="embeddings")
    original_embeddings = Vectors(name='embeddings/glove-sbwc.i25.vec', cache='./embeddings/')
    #torchtext.build_vocab(train, val, test, vectors=vectors)
    print("embeddings loaded")

    data_reader = DataReader(path_to_training, original_embeddings)
    training_instances, training_labels = data_reader.read_data(path_to_training)
    dev_instances, dev_labels = data_reader.read_data(path_to_dev)



    loader_train = DataLoader(training_instances, training_labels)
    loader_dev = DataLoader(dev_instances, dev_labels)

    model = CNN(data_reader.embeddings)
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay = L2)
    #optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_func = nn.NLLLoss(reduction="mean")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader_train.batches(BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            padded_batch = loader_train.pad(batch)
            padded_sequences = padded_batch[0]
            labels = padded_batch[1]
            #print(raw_sequences)
            #padded_sequences = pad_sequence(raw_sequences, batch_first=True)
            #labels = torch.tensor(raw_labels)
            output = model(padded_sequences)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + (loss.item()*BATCH_SIZE)
        print(f"Epoch {epoch}: loss {epoch_loss:0.2f}")
        model.eval()
        correct = 0
        total = 0
        for batch in loader_dev.batches(BATCH_SIZE, shuffle=False):
            #padded_batch = loader_dev.pad(dev)
            padded_batch = loader_train.pad(batch)
            padded_instances = padded_batch[0]
            labels = padded_batch[1]
            #padded_instances = pad_sequence(raw_sequences, batch_first=True)
            #labels = torch.tensor(raw_labels)
            #instances = padded_batch[0]
            #labels = padded_batch[1]
            #print(instances)

            output = model(padded_instances)
            pred = output.argmax(1)
            correct = correct + (pred == labels).sum().item()
            total = total + len(labels)
        accuracy = correct /total
        print(f"Epoch {epoch}: accuracy {accuracy:0.2f}")

#embedding_layer = nn.Embedding.from_pretrained(embedding.vectors)