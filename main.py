from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = "runs"
best_trial = None
best_score = -1
all_results = []

for file in os.listdir(log_dir):
    if not file.startswith("events.out.tfevents"):
        continue

    try:
        file_path = os.path.join(log_dir, file)
        ea = EventAccumulator(file_path)
        ea.Reload()

        scalars = ea.Scalars("eval/f1")
        if scalars:
            final_score = scalars[-1].value  # Get the last eval f1 score
            hparams = {}
            try:
                for k in ea.Tags().get("tensors", []):
                    if k.startswith("hparams/"):
                        tensor_event = ea.Tensors(k)[0]
                        val = tensor_event.tensor_proto.string_val[0].decode("utf-8")
                        hparams[k.split("hparams/")[1]] = val
            except Exception:
                pass

            all_results.append((file, final_score, hparams))

            if final_score > best_score:
                best_score = final_score
                best_trial = (file, final_score, hparams)

    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

print("Best trial:")
print("File:", best_trial[0])
print("F1 Score:", best_trial[1])
print("Hyperparameters:")
for k, v in best_trial[2].items():
    print(f"  {k}: {v}")
