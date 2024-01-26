labels = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder',
          6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist',
          10: 'right_wrist', 11: 'left_hip',
          12: 'right_hip', 13: 'left_knee',
          14: 'right_knee', 15: 'left_ankle',
          16: 'right_ankle'}

body_connections_1 = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
                      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
body_connections_2 = [[1, 2], [5, 6], [5, 7], [6, 8], [9, 11], [10, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
body_connections_3 = body_connections_2 + [[11, 14], [12, 13], [13, 16], [14, 15]]
body_connections_full = [[i, j] for i in range(17) for j in range(17) if i < j]