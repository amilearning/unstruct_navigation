# import torch
# import random

# class ActionSequenceGrid:
#     def __init__(self, num_rows, num_cols):
#         self.num_rows = num_rows
#         self.num_cols = num_cols
#         self.action_sequence_grid = [[torch.tensor([]) for _ in range(num_cols)] for _ in range(num_rows)]
    
#     def add_sample(self, row_index, col_index, action_sample):
#         sample_tensor = torch.tensor(action_sample)
#         self.action_sequence_grid[row_index][col_index] = torch.cat((self.action_sequence_grid[row_index][col_index], sample_tensor.unsqueeze(0)), dim=0)
    
#     def print_grid(self):
#         print("Action Sequence Grid:")
#         for row_index in range(self.num_rows):
#             for col_index in range(self.num_cols):
#                 print(f"Grid [{row_index}][{col_index}]: {len(self.action_sequence_grid[row_index][col_index])}")
    
#     def randomly_pick_samples(self, row_index, col_index, num_samples_to_pick):
#         samples_in_grid = self.action_sequence_grid[row_index][col_index]
#         samples_count = samples_in_grid.size(0)
#         if samples_count > 0:
#             picked_indices = random.sample(range(samples_count), min(num_samples_to_pick, samples_count))
#             picked_samples = samples_in_grid[picked_indices]
#             return picked_samples.tolist()
#         else:
#             return []

# # # Example usage:
# # num_rows = 3
# # num_cols = 4
# # grid = ActionSequenceGrid(num_rows, num_cols)

# # # Simulate adding samples to the grids
# # for row_index in range(num_rows):
# #     for col_index in range(num_cols):
# #         num_samples = random.randint(1, 5)
# #         samples = [random.uniform(0, 1) for _ in range(num_samples)]
# #         for sample in samples:
# #             grid.add_sample(row_index, col_index, sample)

# # # Print the initial grid
# # grid.print_grid()

# # # Randomly pick samples from a specific grid cell
# # row_index = 1
# # col_index = 2
# # num_samples_to_pick = 2
# # randomly_picked_samples = grid.randomly_pick_samples(row_index, col_index, num_samples_to_pick)

# # # Print randomly picked samples
# # print(f"\nRandomly Picked Samples from Grid [{row_index}][{col_index}]:")
# # print(randomly_picked_samples)