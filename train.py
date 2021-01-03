import click

import torch

from tqdm.auto import tqdm
from torchtext import data, datasets
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from classifier import RoBERTaSentimentClassifier
from util import save_checkpoint, save_metrics

@click.command()
@click.argument("file-path", type=click.Path(resolve_path=True))
@click.option("--batch-size", type=int, default=8)
@click.option("--base-model", type=str, default="roberta-base")
@click.option("-e", "--num-epochs", type=int, default=20)
def main(file_path, batch_size, base_model, num_epochs):
    """Train movie sentiment model"""

    print("Initializing models")

    tokenizer = RobertaTokenizerFast.from_pretrained(base_model)
    #model = RobertaForSequenceClassification.from_pretrained("roberta-base")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = RoBERTaSentimentClassifier(device=device, base_model=base_model)

    print(f"Using device {model.device}")

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # set up fields
    TEXT = data.Field(use_vocab=False,
                      include_lengths=False,
                      batch_first=True,
                      lower=False,
                      fix_length=512,
                      tokenize=tokenizer.encode,
                      pad_token=PAD_INDEX,
                      unk_token=UNK_INDEX)

    LABEL = data.Field(sequential=False, use_vocab=True,
                       batch_first=True, dtype=torch.float)

    LABEL.build_vocab()

    print("Load datasets")
    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    test, val = test.split(split_ratio=0.7)

    print("Prepare dataset iterators")
    # make iterator for splits
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=batch_size, device=device)

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    eval_every = len(train_iter) // 2

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("Start training")
    for epoch in range(num_epochs):

        print(f"Epoch {epoch}")

        for (text, labels), _ in tqdm(train_iter):
            labels = labels.type(torch.LongTensor)   
            labels = labels.to(device)  
            output = model(text, labels)

            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (text, label), _ in val_iter:
                        labels = labels.to(device)
                        labels = labels.type(torch.LongTensor)     
                        text = text.to(device)
                        output = model(text, labels)
                        loss, _ = output

                        valid_running_loss += loss.item()

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(val_iter)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                                  average_train_loss, average_valid_loss))

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(file_path + '/' +
                                        'model.pt', model, best_valid_loss)
                        save_metrics(file_path + '/' + 'metrics.pt',
                                     train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
                 valid_loss_list, global_steps_list)
    print('Finished Training!')


if __name__ == "__main__":
    main()
