from app.clustering import best_group, cosine_similarity


def test_cosine_similarity_identity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_best_group_threshold():
    groups = {
        "g1": [[1.0, 0.0], [0.9, 0.1]],
        "g2": [[0.0, 1.0]],
    }
    assert best_group([1.0, 0.0], groups, threshold=0.5) == "g1"
    assert best_group([0.1, 0.1], groups, threshold=0.95) is None
