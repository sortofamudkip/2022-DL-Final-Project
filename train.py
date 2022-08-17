import argparse
from pickletools import optimize
import architecture
from data_loading import load_data
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from utils import WANDB_PROJECT_NAME, get_device, DEFAULT_DATA_PATH, timestamp
import logging

# TODO: Validation?, take X % as validation
# Unet
# Add weights to class imbalance, batch oversampling, make sure that at least 1 sample is in data loaders


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    device,
    num_epochs=25,
    epoch_callback=(lambda epoch: epoch),
):
    wandb.watch(model, log_freq=100)

    running_loss = 0.0
    model.train()

    for epoch in range(num_epochs):
        for input_images, output_labels in train_loader:
            input_images=input_images.to(device)
            output_labels=output_labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(input_images)
                loss = criterion(outputs, output_labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * input_images.size(0)
        epoch_loss = running_loss / len(train_loader)

        wandb.log(
                {"train_loss": running_loss, "epoch": epoch, "epoch_loss": epoch_loss}
            )

        #scheduler.step()

        epoch_callback(epoch)


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with given parameters and save its to some path"
    )
    parser.add_argument(
        "--data_path", default=DEFAULT_DATA_PATH, help="Path to the stored raw data",
    )
    parser.add_argument(
        "--download_data",
        type=bool,
        default=False,
        help="Force downloading the data into the data_path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Number of epochs. Exports one checkpointed model per epoch.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--tags", nargs="+", help="List of tags to find your results in Wandb"
    )
    parser.add_argument(
        "--model_state_file",
        help="Default path where the final model and checkpoints are saved to",
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--model",
        choices=architecture.models_dict.keys(),
        required=True,
        help="Architecture to train. Check architecture.py",
    )
    parser.add_argument(
        "--first_n_rows",
        type=int,
        default=0,
        help="Only use the first N rows of the dataset (for debugging)",
    )
    parser.add_argument(
        "--resume_training",
        default='No',
        help="To determine if model should be trained from scratch or from a checkpoint",
    )
    

    args = parser.parse_args()

    # Setup Wandb with the arguments from ArgumentParser
    wandb.init(project=WANDB_PROJECT_NAME, tags=args.tags)
    wandb.config.update({"phase": "train"})
    wandb.config.update(args)

    # Setup all the things required for training
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    # Lazily setup the model
    input_transforms, model_klass = architecture.models_dict[args.model]
    model = model_klass()
    if args.resume_training=='Yes':
      model.load_state_dict(torch.load(args.model_state_file))
    # Load the training data and apply image transformation and the badge size
    train_loader, _ = load_data(
        args.data_path,
        transforms=input_transforms,
        download=args.download_data,
        batch_size=args.batch_size,
        first_n_rows=args.first_n_rows,
        
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    logging.info("Start training model", model=args.model)

    def epoch_callback(epoch):
        model_state_file = os.path.join(
            args.model_state_path,
            "{}-{}-epoch-{}.pt".format(timestamp(), args.model, epoch),
        )
        logging.info(
            "Saving checkpoint model at epoch",
            model=args.model,
            epoch=epoch,
            model_state_file=model_state_file,
        )

        torch.save(model.state_dict(), model_state_file)
    model=model.to(device)
    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        num_epochs=5,
        epoch_callback=epoch_callback,
    )

    model_state_file = os.path.join(
        args.model_state_path, "{}-{}-final.pt".format(timestamp(), args.model)
    )
    logging.info(
        "Saving final model", model=args.model, model_state_file=model_state_file
    )
    torch.save(model.state_dict(), model_state_file)


if __name__ == "__main__":
    main()
