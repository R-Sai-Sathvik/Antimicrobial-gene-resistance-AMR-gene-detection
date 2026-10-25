"""Unit tests for sequence clustering and leakage detection."""

from src.preprocessing.cluster_splitter import cluster_sequences_minhash


def test_clustering_identical_sequences():
    records = [
        {"id": "seq1", "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGK", "label": 1},
        {"id": "seq2", "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGK", "label": 1},
        {"id": "seq3", "sequence": "ARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR", "label": 0},
    ]
    clusters = cluster_sequences_minhash(records, k=4, identity_threshold=0.70)
    # seq1 and seq2 should belong to the same cluster; seq3 should be in a separate cluster
    assert len(clusters) == 2
    cluster_ids = [[rec["id"] for rec in cl] for cl in clusters]
    assert ["seq1", "seq2"] in cluster_ids or ["seq2", "seq1"] in cluster_ids
