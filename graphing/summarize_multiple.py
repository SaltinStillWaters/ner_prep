import os
import json
import csv

root_dir = "super_out/trials"  # adjust as needed. Should be the dir that contains the multiple model runs. 

output_dir = 'summary'
output_name = 'optuna_all'
output_csv = f'{output_dir}/{output_name}.csv'


results = []

for trial_name in os.listdir(root_dir):
    trial_path = os.path.join(root_dir, trial_name)
    # print(trial_path)
    if not os.path.isdir(trial_path):
        print('not', trial_path)
        continue

    # trainer_state_path = trial_path + '/trainer_state.json'
    # Find checkpoint directories
    checkpoint_dirs = [
        d for d in os.listdir(trial_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(trial_path, d))
    ]
    
    if not checkpoint_dirs:
        print('<--skipped-->', trial_path)
        continue

    # Sort by checkpoint number and pick the highest one
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    last_checkpoint = checkpoint_dirs[-1]
    trainer_state_path = os.path.join(trial_path, last_checkpoint, "trainer_state.json")
    if not os.path.isfile(trainer_state_path):
        continue

    with open(trainer_state_path, "r") as f:
        state = json.load(f)
        trial_params = state.get("trial_params", {})
        best_f1 = state.get("best_metric", None)

        # Find epoch where best_f1 was recorded
        for log_entry in state.get("log_history", []):
            if log_entry.get("eval_f1") == best_f1:
                best_epoch = log_entry.get("epoch")
                break
            
        results.append({
            "per_device_train_batch_size": trial_params.get("per_device_train_batch_size"),
            "learning_rate": trial_params.get("learning_rate"),
            "weight_decay": trial_params.get("weight_decay"),
            "num_train_epochs": trial_params.get("num_train_epochs"),
            "best_epoch": best_epoch,
            "best_f1": best_f1,
        })


with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["per_device_train_batch_size", "learning_rate", "weight_decay", "num_train_epochs", "best_epoch", "best_f1"])
    writer.writeheader()
    writer.writerows(results)