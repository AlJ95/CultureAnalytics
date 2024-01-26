import numpy as np

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors.
    :param vec1: vector 1
    :param vec2: vector 2
    :return: angle in degrees
    """
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return 0 if np.isnan(np.degrees(angle)) else np.degrees(angle)


def calculate_unit_circle_pos(vec1, vec2):
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    angle = np.array([np.cos(angle), np.sin(angle)])

    if any(np.isnan(angle)):
        angle = np.zeros((2,))

    return angle
