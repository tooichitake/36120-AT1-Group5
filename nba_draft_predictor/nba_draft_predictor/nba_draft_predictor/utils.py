import pandas as pd

def generate_submission(player_ids, predictions, filename="submission.csv"):
    submission = pd.DataFrame({
        "player_id": player_ids,
        "drafted": predictions
    })
    submission.to_csv(filename, index=False)
