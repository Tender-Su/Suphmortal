import torch
import numpy as np

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False, label_smoothing=0.0):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval() if grp is not None else None
        self.grp_dtype = next(self.grp.parameters()).dtype if self.grp is not None else torch.float64
        self.uniform_init = uniform_init
        # Historical cheap smoothing fallback for terminal rank labels.
        # This is not the mainline RVR-style variance-reduction implementation.
        self.label_smoothing = label_smoothing

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=self.grp_dtype, device=self.device)

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], dtype=self.grp_dtype, device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob_from_matrix(self, matrix, player_id, rank_by_player):
        eps = self.label_smoothing
        final_ranking = torch.zeros((1, 4), dtype=self.grp_dtype, device=self.device)
        if eps > 0:
            final_ranking.fill_(eps / 4)
            final_ranking[0, rank_by_player[player_id]] = 1.0 - 3 * eps / 4
        else:
            final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)
        return self.calc_rank_prob_from_matrix(matrix, player_id, rank_by_player)

    def calc_delta_pt_all_players(self, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)
        exp_pts_by_player = []
        for player_id in range(4):
            rank_prob = self.calc_rank_prob_from_matrix(matrix, player_id, rank_by_player)
            exp_pts_by_player.append(rank_prob @ self.pts)
        exp_pts = torch.stack(exp_pts_by_player, dim=-1)
        reward = exp_pts[1:] - exp_pts[:-1]
        reward_np = reward.cpu().numpy()
        return np.asarray(reward_np, dtype=np.float32)

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        reward_np = self.calc_delta_pt_all_players(grp_feature, rank_by_player)
        return reward_np[:, player_id]
