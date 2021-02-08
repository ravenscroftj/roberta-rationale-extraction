#%%
import click
import torch
import dill
from typing import Union
from tqdm.auto import tqdm
from torchtext import data, datasets
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from classifier import RoBERTaSentimentClassifier
from util import save_checkpoint, save_metrics, save_cached_dataset, load_cached_dataset
from pathlib import Path


@click.command()
@click.argument("file-path", type=click.Path(resolve_path=True))
@click.option("--batch-size", type=int, default=8)
@click.option("--base-model", type=str, default="roberta-base")
@click.option("-e", "--num-epochs", type=int, default=20)
def main(file_path, batch_size, base_model, num_epochs):
    """Train movie sentiment model"""


    # %%
    # base_model  = "roberta-base"
    # batch_size=8
    # num_epochs=5
    print("Initializing models")

    tokenizer = RobertaTokenizerFast.from_pretrained(base_model)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = RoBERTaSentimentClassifier(device=device, base_model=base_model)

    print(f"Using device {model.device}")

    #%%
    train_cache = Path(".data/cache/train_data")
    val_cache = Path(".data/cache/validate_data")

    if train_cache.exists() and val_cache.exists():
        print("Load cached datasets")
        train = load_cached_dataset(train_cache)
        val = load_cached_dataset(val_cache)
    else:
        print("Generating datasets")
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

        LABEL = data.LabelField()

        # make splits for data
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        LABEL.build_vocab(train)

        test, val = test.split(split_ratio=0.9)

        print("Cache train and validate sets")

        save_cached_dataset(train,train_cache)
        save_cached_dataset(val, val_cache)
        
    print("Prepare dataset iterators")
    # make iterator for splits
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=batch_size, device=device)

    #%%
    for batch in val_iter:
        if batch.text.shape[0] != batch.label.shape[0]:
            print(batch)
        # print(batch.text.shape, batch.label.shape)
        # break
    #%%
    #dir(val_iter)
    #%%
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    best_valid_loss = float("Inf")

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for item in train_iter:
        print(item)
        break

    print("Start training")
    for epoch in range(1,num_epochs+1):

        print(f"Epoch {epoch}")

        train_iter.init_epoch()
        val_iter.init_epoch()

        for i, (text, labels) in enumerate(tqdm(train_iter, desc="train")):
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


        model.eval()
        with torch.no_grad():

            answers = []

            # validation loop
            for i, (text, labels) in enumerate(tqdm(val_iter, desc="validate")):
                labels = labels.type(torch.LongTensor)     
                labels = labels.to(device)
                output = model(text, labels)
                loss, preds = output
                
                correct = torch.argmax(preds, dim=1) == labels

                answers.extend(correct.cpu().tolist())


                valid_running_loss += loss.item()


            average_accuracy = sum([1 for a in answers if a]) / len(answers)

                

            # evaluation
            average_train_loss = running_loss / epoch
            average_valid_loss = valid_running_loss / 10
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            global_steps_list.append(global_step)

            # resetting running values
            running_loss = 0.0
            valid_running_loss = 0.0
            model.train()

            # print progress
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                            average_train_loss, average_valid_loss, average_accuracy))

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
