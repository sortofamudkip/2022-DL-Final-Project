import argparse
from pickletools import optimize
import architecture
from data_loading import load_data,load_data_test
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from utils import WANDB_PROJECT_NAME, get_device, DEFAULT_DATA_PATH
import sklearn.metrics
import numpy as np
import csv
# TODO: divide by total
# TODO: Use actual validation
# TODO: Submit to kaggle
# TODO: http://gradcam.cloudcv.org/
# TODO: Posthoc Captum


def test(model, test_loader, device,mode='val'):
  if mode =='val':
      wandb.watch(model, log_freq=100)

      test_size = len(test_loader.dataset)
      batch_size = test_loader.batch_size
      running_accuracy = 0

      y_true = np.empty(test_size)
      y_predicated = np.empty(test_size)
      with torch.no_grad():
          for i, (input_images, output_labels) in enumerate(test_loader):
              input_images.to(device)
              output_labels.type(torch.LongTensor).to(device)

              predicted_outputs = model(input_images)
              _, predicted = torch.max(predicted_outputs, 1)
              y_true[i * batch_size : (i + 1) * batch_size] = output_labels
              y_predicated[i * batch_size : (i + 1) * batch_size] = predicted

              # there really isn't a point for this but let's try it anyway
              running_accuracy += (predicted == output_labels).sum().item()
              wandb.log({"accuracy": running_accuracy})

      f1 = sklearn.metrics.f1_score(y_true, y_predicated)
      recall = sklearn.metrics.recall_score(y_true, y_predicated)
      precision = sklearn.metrics.precision_score(y_true, y_predicated)
      auc = sklearn.metrics.roc_auc_score(y_true, y_predicated)
      accuracy = sklearn.metrics.accuracy_score(y_true, y_predicated)

      # OR to log a final metric at the end of training you can also use wandb.summary

      # https://docs.wandb.ai/guides/integrations/scikit
      wandb.run.summary["accuracy"] = accuracy
      wandb.run.summary["f1"] = f1
      wandb.run.summary["recall"] = recall
      wandb.run.summary["precision"] = precision
      wandb.run.summary["auc"] = auc
      wandb.sklearn.plot_roc(y_true, y_predicated, [0, 1])
      wandb.log(
          {
              "conf_matrix": wandb.plot.confusion_matrix(
                  probs=None, y_true=y_true, preds=y_predicated, class_names=[0, 1],
              )
          }
      )
  elif mode=='test':
    wandb.watch(model, log_freq=100)

      test_size = len(test_loader.dataset)
      batch_size = test_loader.batch_size
      running_accuracy = 0

      y_true = np.empty(test_size)
      y_predicated = np.empty(test_size)
      f = open('results.csv', 'w')
      writer = csv.writer(f)
      header = ['id', 'label']
      writer.writerow(header)
      with torch.no_grad():
          for i, (input_images, output_labels) in enumerate(test_loader):
              input_images.to(device)
              output_labels.type(torch.LongTensor).to(device)

              predicted_outputs = model(input_images)
              _, predicted = torch.max(predicted_outputs, 1)
              row=[output_labels,predicted[0].item()]
              writer.writerow(row)
              y_true[i * batch_size : (i + 1) * batch_size] = output_labels
              y_predicated[i * batch_size : (i + 1) * batch_size] = predicted

              # there really isn't a point for this but let's try it anyway
              running_accuracy += (predicted == output_labels).sum().item()
              wandb.log({"accuracy": running_accuracy})
      f.close()
      os.system('kaggle competitions submit histopathologic-cancer-detection -f results.csv -m "submission"')
      f1 = sklearn.metrics.f1_score(y_true, y_predicated)
      recall = sklearn.metrics.recall_score(y_true, y_predicated)
      precision = sklearn.metrics.precision_score(y_true, y_predicated)
      auc = sklearn.metrics.roc_auc_score(y_true, y_predicated)
      accuracy = sklearn.metrics.accuracy_score(y_true, y_predicated)

      # OR to log a final metric at the end of training you can also use wandb.summary

      # https://docs.wandb.ai/guides/integrations/scikit
      wandb.run.summary["accuracy"] = accuracy
      wandb.run.summary["f1"] = f1
      wandb.run.summary["recall"] = recall
      wandb.run.summary["precision"] = precision
      wandb.run.summary["auc"] = auc
      wandb.sklearn.plot_roc(y_true, y_predicated, [0, 1])
      wandb.log(
          {
              "conf_matrix": wandb.plot.confusion_matrix(
                  probs=None, y_true=y_true, preds=y_predicated, class_names=[0, 1],
              )
          }
      )



def main():
    parser = argparse.ArgumentParser(description="Test an imported model")
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help="Path to the stored raw data. Downloads the data if it cannot be found.",
    )
    parser.add_argument(
        "--download_data",
        type=bool,
        default=False,
        help="Force downloading the data into the data_path",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument(
        "--model_state_file", required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--tags", nargs="+", help="List of tags to find your results in Wandb"
    )
    parser.add_argument(
        "--model",
        choices=architecture.models.keys(),
        required=True,
        help="Architecture to test. Check architecture.py",
    )
    parser.add_argument(
        "--first_n_rows",
        type=int,
        default=0,
        help="Only use the first N rows of the dataset (for debugging)",
    )
    parser.add_argument(
        "--mode",
        default='val',
        help="To determine if model should be tested on validation or test data",
    )

    args = parser.parse_args()

    wandb.init(project=WANDB_PROJECT_NAME, tags=args.tags)
    wandb.config.update({"phase": args.mode})
    wandb.config.update(args)

    device = get_device()
    input_transforms, model_klass = architecture.models[args.model]
    model = model_klass()
    model.load_state_dict(torch.load(args.model_state_file))
    if args.mode =='val':
      _, test_loader = load_data(
          args.data_path,
          transforms=input_transforms,
          download=args.download_data,
          batch_size=args.batch_size,
          first_n_rows=args.first_n_rows,
      )
    else:
      test_loader = load_data_test(
          args.data_path,
          transforms=input_transforms,
          batch_size=args.batch_size,
          
      )

    model.eval()

    test(model, test_loader, device=device)


if __name__ == "__main__":
    main()
