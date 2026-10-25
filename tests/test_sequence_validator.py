"""Unit tests for sequence validation and alphabet checking."""

from src.preprocessing.sequence_validator import SequenceValidator


def test_valid_sequence():
    validator = SequenceValidator(min_len=5, max_len=100)
    valid_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGK"
    is_valid, msg = validator.validate_single_sequence(valid_seq)
    assert is_valid
    assert msg == "Valid"


def test_invalid_length_short():
    validator = SequenceValidator(min_len=20, max_len=50)
    short_seq = "MSKGEEL"
    is_valid, msg = validator.validate_single_sequence(short_seq)
    assert not is_valid
    assert "min_length" in msg


def test_invalid_characters():
    validator = SequenceValidator(min_len=5, max_len=50)
    invalid_seq = "MSKGEELFTGZ123!"
    is_valid, msg = validator.validate_single_sequence(invalid_seq)
    assert not is_valid
    assert "Invalid amino acid" in msg
