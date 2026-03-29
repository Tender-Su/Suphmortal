import torch
import glob

def test_greedy_baseline():
    files = glob.glob('D:/mahjong_data/grp_pt/val/*.pt')
    if not files:
        print("No validation files found.")
        return

    data = torch.load(files[0], weights_only=False)

    total_steps = 0
    correct_predictions = 0

    for feature, final_rank in data:
        # final_rank is e.g. (2, 1, 0, 3) which means player 0 is rank 2, player 1 is rank 1, player 2 is rank 0 (1st), player 3 is rank 3
        # In GRP model, rank is 0-indexed. 0 means 1st place, 3 means 4th place.
        # feature has shape [seq_len, 7].
        # feature[:, 3:7] are the scores of the 4 players at each step.

        seq_len = feature.shape[0]

        for i in range(seq_len):
            current_scores = feature[i, 3:7]
            # Predict rank by sorting current scores in descending order.
            # argsort(descending) gives the player IDs ordered by rank (1st to 4th)
            # We want to map player ID -> rank.
            # e.g. scores = [2000, 3000, 4000, 1000]
            # sorted_players = [2, 1, 0, 3] (player 2 is 1st, player 1 is 2nd...)
            # To get rank_by_player: player 0 is rank 2, player 1 is rank 1...

            sorted_players = torch.argsort(current_scores, descending=True)
            predicted_rank = torch.zeros(4, dtype=torch.int64)
            for rank, player_id in enumerate(sorted_players):
                predicted_rank[player_id] = rank

            predicted_rank = tuple(predicted_rank.tolist())

            if predicted_rank == final_rank:
                correct_predictions += 1
            total_steps += 1

    acc = correct_predictions / total_steps
    print(f"Greedy Baseline Accuracy (sorting current scores): {acc:.4f} ({correct_predictions}/{total_steps})")

if __name__ == "__main__":
    test_greedy_baseline()
